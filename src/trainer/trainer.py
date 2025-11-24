"""
@author : seauagain
@date : 2025.11.01 
"""


import os 
import sys 
import time 
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
    def __init__(self, config, logger=None) -> None:
        self.config = config 
        self.device = config.device
        self.rank = 0
        self.world_size = 1 
        self.use_ddp = False 
        self.network = None  # -> self.build_network()
        self.was_initialized = False
        self.model_root = config.model_root
        self.model_name = config.model_name

        self.logger = logger

    def initialize(self, config):
        """
        Initialize the network architechture, DDP configuration, 
        """
        def init_weights(module):
                """
                set the initialization for netowrks parameters.
                """
                gamma = 1
                if isinstance(module , (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
                    nn.init.normal_(module.weight, mean=0, std=module.weight.size(1)**(-gamma) )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        if not self.was_initialized:
            self.network = self.build_network(config) # build network
            self.network.apply(init_weights)    # weights init
            self.was_initialized = True
    
    def build_network(self, config):
        """create networks"""
        from src.model import Transformer
        network = Transformer(
                            en_vocab_size=config.en_vocab_size,
                            de_vocab_size=config.zh_vocab_size,
                            d_model=config.d_model,
                            num_heads=config.nums_heads,
                            num_layers=config.num_layers,
                            d_ff=config.d_ff,
                            max_seq_length=config.max_seq_length,
                            dropout=config.dropout
                            )
        return network 

    def _count_parameters(self):
        """"""
        return sum(p.numel() for p in self.network.parameters())

    def setup_ddp(self, rank, world_size, backend="nccl"):
        """initialize the environment for DDP"""
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
        """clean up the settings for DDP"""
        if self.use_ddp:
            dist.destroy_process_group()

    def build_loss(self, config):
        """loss function"""    
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.trg_pad_idx)
    
    def build_optimizer(self, config):
        """the optimizer and lr scheduler"""
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.init_lr) 
        self.lr_scheduler = self.warmup_scheduler(self.optimizer, config.warmup_epochs, config.max_epochs)

    def warmup_scheduler(self, optimizer, warmup_epochs, max_epoch):
        """创建带warmup的学习率调度器"""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (max_epoch - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return LambdaLR(optimizer, lr_lambda)

    def training_entrance(self, config):
        if int(os.environ.get("WORLD_SIZE", "1")) > 1: 
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
            train_losses.append(train_loss)

            # validation
            if epoch % config.validloss_interval == 0 or epoch == config.max_epochs:
                val_loss = self.validate_epoch(val_loader, config)
                val_losses.append(val_loss)
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f'Epoch: {epoch}/{config.max_epochs}{"":^2} | Rank: {self.rank:^2} | Train loss: {train_loss:.5f} | Valid loss: {val_loss:.5f} | Time: {epoch_time:.2f}s', process="all")
            else:
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f'Epoch: {epoch}/{config.max_epochs}{"":^2} | Rank: {self.rank:^2} | Train loss: {train_loss:.5f} | Time: {epoch_time:.2f}s', process="all")
            
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
        """train for one epoch"""
        self.network.train()
        total_loss = 0
        num_batches = 0
        for _, (src, trg) in enumerate(dataloader):
            src, trg = src.to(self.device), trg.to(self.device)
            # trg_input: 去除句子末尾
            trg_input = trg[:, :-1]
            # trg_output: 去除句子开头
            trg_output = trg[:, 1:].contiguous().view(-1)
            self.optimizer.zero_grad()
            output = self.network(src, trg_input, config.src_pad_idx, config.trg_pad_idx)
            output = output.contiguous().view(-1, output.size(-1))
            loss = self.criterion(output, trg_output)
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader, config):
        """valiadation for one epoch"""
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
        pass

    def plot_loss(self, train_losses, val_losses, save_path='loss_plot.png'):
        if not self.use_ddp or self.rank == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss', linewidth=2)
            if val_losses:
                plt.plot(val_losses, label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_checkpoint(self, epoch, train_losses, val_losses):
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
