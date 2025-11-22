"""
@author : seauagain
@date : 2025.11.01 
"""


import os 
import sys 
import time 
from typing import Optional, Dict 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader, DistributedSampler 
from torch.optim.lr_scheduler import LambdaLR

class Trainer:
    def __init__(self, config) -> None:
        self.config = config 
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.rank = 0
        self.world_size = 1 
        self.is_ddp = False 
        self.model = None 
    
    def setup_ddp(self, rank, world_size, backend="nccl"):
        """initialize the environment for DDP"""
        self.rank = rank 
        self.world_size = world_size
        self.is_ddp = True

        dist.initz_process_group(backend=backend, rank=rank, world_size = world_size)
        torch.cuda.set_device(rank)
        self.device = torch.device(f"cuda:{rank}")
        self.model = self.model.to(self.device)
        self.model = DDP(self.mode, device_ids = [rank])
    
    def cleanup_ddp():
        """clean up the settings for DDP"""
        if self.is_ddp:
            dist.destroy_process_group()

    def create_dataloader(self, dataset, batch_size, shuffle=True, num_workers=4):
        """create the training dataloader for DDP"""

        if self.is_ddp:
            sampler = DistributedSampler(dataset, num_workers=self.world_size, rank=self.rank, shuffle=shuffle)
            shuffle = False #
        
        else:
            sampler = None
            
        return DataLoader(
                        dataset, 
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True,
                        sampler=sampler
                    )

    def warmup_scheduler(self, optimizer, warmup_epochs, total_epochs):
        """创建带warmup的学习率调度器"""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return LambdaLR(optimizer, lr_lambda)
    

    def warmup_scheduler(self, optimizer, warmup_epochs, total_epochs):
        """创建带warmup的学习率调度器"""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                
            total_loss += loss.item()
            num_batches += 1
            
            # 每100个batch打印一次
            if batch_idx % 100 == 0 and (not self.is_ddp or self.rank == 0):
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self, dataloader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def plot_loss(self, train_losses, val_losses, save_path='loss_plot.png'):
        """可视化loss曲线"""
        if not self.is_ddp or self.rank == 0:
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
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, train_losses, val_losses, path):
        """保存检查点"""
        if not self.is_ddp or self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if self.is_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': self.config
            }
            torch.save(checkpoint, path)
    
    def run_training(self, train_dataset, val_dataset, config):
        """单卡训练"""
        self.model = self.model.to(self.device)
        
        # 创建DataLoader
        train_loader = self.create_dataloader(train_dataset, config['batch_size'])
        val_loader = self.create_dataloader(val_dataset, config['batch_size'], shuffle=False) if val_dataset else None
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()
        
        # 学习率调度器
        scheduler = None
        if config.get('use_warmup', True):
            scheduler = self.warmup_scheduler(optimizer, config.get('warmup_epochs', 5), config['epochs'])
        
        train_losses, val_losses = [], []
        
        print(f"开始训练，设备: {self.device}")
        start_time = time.time()
        
        for epoch in range(config['epochs']):
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            train_losses.append(train_loss)
            
            # 验证
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader, criterion)
                val_losses.append(val_loss)
            
            # 打印进度
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{config["epochs"]}, Train Loss: {train_loss:.4f},'
                  f'Val Loss: {val_loss:.4f}, Time: {elapsed_time:.1f}s')
            
            # 保存检查点
            if (epoch + 1) % config.get('save_freq', 10) == 0:
                self.save_checkpoint(epoch, self.model, optimizer, scheduler, train_losses, val_losses, 
                                   f'checkpoint_epoch_{epoch+1}.pth')
        
        # 最终可视化
        self.plot_loss(train_losses, val_losses)
        print("训练完成！")
        
        return train_losses, val_losses
    
    def run_ddp_training(self, rank, world_size, train_dataset, val_dataset, config):
        """DDP多卡训练"""
        self.setup_ddp(rank, world_size)
        
        try:
            # 创建DataLoader（DDP会自动处理分布式采样）
            train_loader = self.create_dataloader(train_dataset, config['batch_size'])
            val_loader = self.create_dataloader(val_dataset, config['batch_size'], shuffle=False) if val_dataset else None
            
            # 优化器和损失函数
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
            criterion = nn.MSELoss()
            
            # 学习率调度器
            scheduler = None
            if config.get('use_warmup', True):
                scheduler = self.warmup_scheduler(optimizer, config.get('warmup_epochs', 5), config['epochs'])
            
            train_losses, val_losses = [], []
            
            if rank == 0:
                print(f"开始DDP训练，世界大小: {world_size}")
            start_time = time.time()
            
            for epoch in range(config['epochs']):
                # 设置epoch给sampler
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                # 训练
                train_loss = self.train_epoch(train_loader, optimizer, criterion, scheduler)
                train_losses.append(train_loss)
                
                # 验证
                val_loss = None
                if val_loader:
                    val_loss = self.validate(val_loader, criterion)
                    val_losses.append(val_loss)
                
                # 只在rank 0打印和保存
                if rank == 0:
                    elapsed_time = time.time() - start_time
                    print(f'Epoch {epoch+1}/{config["epochs"]}, Train Loss: {train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Time: {elapsed_time:.1f}s')
                    
                    # 保存检查点
                    if (epoch + 1) % config.get('save_freq', 10) == 0:
                        self.save_checkpoint(epoch, self.model, optimizer, scheduler, train_losses, val_losses, 
                                           f'checkpoint_epoch_{epoch+1}.pth')
            
            # 最终可视化（只在rank 0）
            if rank == 0:
                self.plot_loss(train_losses, val_losses)
                print("DDP训练完成！")
                
        finally:
            self.cleanup_ddp()
            
        return train_losses, val_losses
    