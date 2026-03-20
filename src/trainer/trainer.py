"""
@author : seauagain
@date : 2025.11.01 
"""


import os
import sys
import time
import math
import json
import logging
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from src.data.dataloader import get_train_val_loader

class Trainer:
    """Manages the full training lifecycle: initialization, DDP setup, training loop,
    validation, checkpointing, and loss logging."""

    def __init__(self, config, logger=None) -> None:
        """
        Args:
            config: configuration object with all hyperparameters and paths.
            logger: optional distributed-aware logger instance.
        """
        self.config = config
        self.device = config.device
        self.rank = 0
        self.world_size = 1
        self.use_ddp = False
        self.network = None
        self.was_initialized = False
        self.model_root = config.model_root
        self.model_name = config.model_name
        self.logger = logger

    def initialize(self, config):
        """Build the network and apply weight initialization (idempotent).

        Args:
            config: configuration object passed to build_network.
        """
        def init_weights(module):
            """Initialize Linear layer weights with scaled normal distribution."""
            gamma = 1
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0, std=module.weight.size(1)**(-gamma))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if not self.was_initialized:
            self.network = self.build_network(config)
            self.network.apply(init_weights)
            self.was_initialized = True

    def build_network(self, config):
        """Instantiate the Transformer model from config hyperparameters.

        Args:
            config: must have en_vocab_size, zh_vocab_size, d_model, nums_heads,
                    num_layers, d_ff, max_seq_length, dropout.
        Returns:
            Transformer model (on CPU; moved to device later).
        """
        from src.model import Transformer
        return Transformer(
            en_vocab_size=config.en_vocab_size,
            de_vocab_size=config.zh_vocab_size,
            d_model=config.d_model,
            num_heads=config.nums_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout
        )

    def _count_parameters(self):
        """Return total number of trainable parameters in the network."""
        return sum(p.numel() for p in self.network.parameters())

    def setup_ddp(self, rank, world_size, backend="nccl"):
        """Initialize DistributedDataParallel for multi-GPU training.

        Args:
            rank: this process's rank (0-indexed).
            world_size: total number of processes.
            backend: communication backend (default: "nccl" for GPU).
        """
        self.rank = rank 
        self.world_size = world_size
        self.use_ddp = True

        dist.init_process_group(backend=backend, rank=rank, world_size = world_size)
        torch.cuda.set_device(rank)
        self.device = torch.device(f"cuda:{rank}")
        self.network = self.network.to(self.device)
        self.network = DDP(self.network, device_ids = [rank])
        self.logger.info(f"Process {rank}: Model initialized with DDP", process="all")
    
    def cleanup_ddp(self):
        """Destroy the DDP process group after training completes."""
        if self.use_ddp:
            dist.destroy_process_group()

    def build_loss(self, config):
        """Instantiate the loss function.

        Uses CrossEntropyLoss with padding index ignored so pad tokens
        don't contribute to the loss.

        Args:
            config: must have trg_pad_idx.
        """
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.trg_pad_idx)
    
    def build_optimizer(self, config):
        """Instantiate Adam optimizer and warmup+cosine lr scheduler.

        Args:
            config: must have init_lr, warmup_epochs, max_epochs.
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.init_lr) 
        self.lr_scheduler = self.warmup_scheduler(self.optimizer, config.warmup_epochs, config.max_epochs)

    def warmup_scheduler(self, optimizer, warmup_epochs, max_epoch):
        """Create a learning rate scheduler with linear warmup and cosine decay.

        During warmup (epoch < warmup_epochs): lr scales linearly from 0 to init_lr.
        After warmup: lr follows a cosine annealing curve down to 0.

        Args:
            optimizer: the optimizer whose lr will be scheduled.
            warmup_epochs: number of epochs for linear warmup.
            max_epoch: total number of training epochs.
        Returns:
            LambdaLR scheduler (epoch-level, call .step() once per epoch).
        """
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (max_epoch - warmup_epochs)
                # Use math.cos (returns a Python float) instead of torch.cos
                # to avoid LambdaLR receiving a 0-dim tensor multiplier.
                return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def training_entrance(self, config):
        """Entry point that auto-detects DDP vs single-GPU and dispatches accordingly.

        Reads WORLD_SIZE from environment (set by torchrun) to decide whether to
        initialize DDP. Cleans up DDP after training completes.

        Args:
            config: training configuration object.
        """
            # to avoid the case: torchrun --nproc-per-node=1
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            self.use_ddp = True
            self.setup_ddp(rank, world_size)
            self.run_training(config)
            self.cleanup_ddp()
        else:
            self.run_training(config)
    
    def run_training(self, config):
        """training for both single-GPU and multi-GPUs."""
        self.network = self.network.to(self.device) # move to device


        ########################### user configuration ###################################
        train_loader, val_loader, en_vocab, zh_vocab, special_tokens = get_train_val_loader(config.train_data_path, batch_size=config.batch_size, val_split=config.valid_ratio, ddp=self.use_ddp, rank=self.rank, world_size=self.world_size)
        ########################### user configuration ###################################
        
        # 优化器、损失函数与学习率调度器
        self.build_loss(config)
        self.build_optimizer(config)

        train_losses, val_losses = [], []

        # Save initial model state (only on rank 0)
        if self.rank == 0:
            self.save_checkpoint(0, train_losses, val_losses)
        if self.use_ddp:
            dist.barrier()

        start_time = time.time()
        
        for epoch in range(1, config.max_epochs + 1):
            if self.use_ddp:
                train_loader.sampler.set_epoch(epoch) # reset sampler seed. 

            # training
            epoch_start_time = time.time()
            train_loss = self.train_epoch(train_loader, config)

            train_losses.append( [epoch, train_loss] )

            # Step lr scheduler once per epoch (warmup_scheduler uses epoch-level semantics)
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # validation
            if epoch % config.validloss_interval == 0 or epoch == config.max_epochs:
                val_loss = self.validate_epoch(val_loader, config)
                val_losses.append( [epoch, val_loss] )
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f'Epoch: {epoch}/{config.max_epochs}{"":^2} | Rank: {self.rank:^2} | Train loss: {train_loss:.5f} | Valid loss: {val_loss:.5f} | Time: {epoch_time:.2f}s', process="all")
            else:
        
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f'Epoch: {epoch}/{config.max_epochs}{"":^2} | Rank: {self.rank:^2} | Train loss: {train_loss:.5f} | Time: {epoch_time:.2f}s', process="all")
            
            ## save 
            if epoch % config.saveloss_interval == 0 and (not self.use_ddp or self.rank == 0):
                self.save_loss(train_losses, val_losses)
                self.plot_loss(train_losses, val_losses)
            
            if epoch % config.saveckpt_interval == 0 and (not self.use_ddp or self.rank == 0):
                self.save_checkpoint(epoch, train_losses, val_losses)
        

        if not self.use_ddp or self.rank == 0:
            # Save final model and loss
            self.save_loss(train_losses, val_losses)
            self.plot_loss(train_losses, val_losses)
            self.save_checkpoint(epoch, train_losses, val_losses)
            timecost = time.time() - start_time
            self.logger.info(f"Training completed successfully, total time cost: {timecost/3600:.2f} h")

    def train_epoch(self, dataloader, config):
        """Train for one epoch.

        Performs forward pass, loss computation, and backprop for every batch.
        Note: lr_scheduler.step() is intentionally NOT called here — it is called
        once per epoch in run_training() because warmup_scheduler uses epoch-level
        semantics (stepping per batch would exhaust all epochs in the first epoch).

        Args:
            dataloader: training DataLoader yielding (src, trg) batches.
            config: training configuration with src_pad_idx, trg_pad_idx.
        Returns:
            Average training loss over all batches.
        """
        self.network.train()
        total_loss = 0
        num_batches = 0
        for _, (src, trg) in enumerate(dataloader):
            src, trg = src.to(self.device), trg.to(self.device)
            # Teacher forcing: feed trg[:-1] as input, predict trg[1:] as output
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:].contiguous().view(-1)
            self.optimizer.zero_grad()
            output = self.network(src, trg_input, config.src_pad_idx, config.trg_pad_idx)
            output = output.contiguous().view(-1, output.size(-1))
            loss = self.criterion(output, trg_output)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader, config):
        """Run validation for one epoch (no gradient updates).

        Args:
            dataloader: validation DataLoader yielding (src, trg) batches.
            config: must have src_pad_idx, trg_pad_idx.
        Returns:
            Average validation loss over all batches.
        """
        self.network.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for src, trg in dataloader:
                src = src.to(self.device)
                trg = trg.to(self.device)

                trg_input = trg[:, :-1]
                trg_output = trg[:, 1:].contiguous().view(-1)

                output = self.network(src, trg_input, config.src_pad_idx, config.trg_pad_idx)
                output = output.contiguous().view(-1, output.size(-1))
                loss = self.criterion(output, trg_output)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches
    
    def save_loss(self, train_losses, val_losses):
        """Save train and validation losses to a JSON file.

        Args:
            train_losses: list of [epoch, loss] pairs from training.
            val_losses: list of [epoch, loss] pairs from validation.
        """
        loss_path = os.path.join(self.model_root, self.model_name, "losses.json")
        with open(loss_path, 'w') as f:
            json.dump({'train': train_losses, 'val': val_losses}, f)

    def plot_loss(self, train_losses, val_losses, save_path='loss_plot.png'):
        """Plot and save training/validation loss curves.

        Args:
            train_losses: list of [epoch, loss] pairs.
            val_losses: list of [epoch, loss] pairs.
            save_path: file path for the saved PNG figure.
        """
        if not self.use_ddp or self.rank == 0:
            plt.figure(figsize=(10, 6))

            if train_losses:
                train_steps = [x[0] for x in train_losses]             # 提取 steps
                train_vals = [x[1] for x in train_losses]              # 提取 loss values
                plt.plot(train_steps, train_vals, label='Training Loss', linewidth=2)
            if val_losses:
                val_steps = [x[0] for x in val_losses]
                val_vals = [x[1] for x in val_losses]
                plt.plot(val_steps, val_vals, label='Validation Loss', linewidth=2)
                
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_checkpoint(self, epoch, train_losses, val_losses):
        """Save model weights, optimizer state, scheduler state, and losses to disk.

        Only rank 0 writes the checkpoint in DDP mode.

        Args:
            epoch: current epoch number (used in the filename).
            train_losses: list of [epoch, loss] pairs accumulated so far.
            val_losses: list of [epoch, loss] pairs accumulated so far.
        """
        ckpt_path = os.path.join(self.model_root, self.model_name, f"ckpt{epoch}.pth")
        if not self.use_ddp or self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.network.module.state_dict() if self.use_ddp else self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': self.config.__dict__
            }
            torch.save(checkpoint, ckpt_path)
    



'''deprecated code

    @staticmethod
    def find_free_port():
        """find a port available for DDP"""
        import socket 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def train_epoch(self, dataloader):
        """train for one epoch"""
        self.network.train()
        total_loss = 0
        num_batches = 0
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader, criterion):
        """valiadation for one epoch"""
        self.network.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches


'''
