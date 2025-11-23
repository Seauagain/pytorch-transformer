"""
@author : seauagain
@date : 2025.11.01 
"""

import torch
from src.trainer import Trainer
from src.utils import logging_args, setup_logging

from pyinstrument import Profiler

def timecost(func):
    """performance analysis tool"""
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        res = func(*args, **kwargs)
        profiler.stop()
        profiler.print()
        return res

    return wrapper

class DataAttr(dict):
    """dict['key'] to dict.key"""
    def __getattr__(self, item):
        return self[item] 

@timecost
def main():
    # 配置参数
    config = {
        'lr': 1e-3,
        'batch_size': 32,
        'max_epochs': 3,
        'warmup_epochs': 5,
        'use_warmup': True,
        'save_freq': 10,
        'device': 'cuda:0',
        "seed": 42,
        "model_root": "results",
        "model_name": "test_transformer",
        "current_time": "",
        "validloss_interval":10,
        "saveloss_interval":10,
        "saveckpt_interval":100,

    }
    config = DataAttr(config)
    config.en_vocab_size = 70608
    config.zh_vocab_size =  5350
    config.d_model = 512
    config.d_ff = 1024
    config.max_seq_length = 5000
    config.dropout = 0.1
    config.nums_heads = 8
    config.num_layers = 6

    config.train_data_path = "./dataset/translation2019zh_train50k.json"
    config.split_ratio = 0.1
    config.init_lr = 1e-4
    # config.max_epoch = 

    from src.data.dataloader import get_train_val_loader
    train_loader, val_loader, en_vocab, zh_vocab, special_tokens = get_train_val_loader(config.train_data_path, batch_size=config.batch_size, val_split=config.split_ratio)
    config.src_pad_idx = special_tokens["src_pad_idx"]
    config.trg_pad_idx = special_tokens["trg_pad_idx"]
    config.trg_bos_idx = special_tokens["trg_bos_idx"]
    config.trg_eos_idx = special_tokens["trg_eos_idx"]


    import os 
    model_dir = os.path.join(config.model_root, config.model_name)
    os.makedirs(model_dir, exist_ok=True)

    setup_logging(config)
    logging_args(config)  ## print hyper-parameters in log file
    trainer = Trainer(config)
    trainer.initialize(config)
    trainer.training_entrance(config)


if __name__ == '__main__':
    main()

    ## torchrun --nprocnode 2 train.py
    ## python train.py