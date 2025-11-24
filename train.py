"""
@author : seauagain
@date : 2025.11.01 
"""
import os, sys, time 
import torch
from src.config import default_parser
from src.utils import initial_distributed_logger, setup_current_time, setup_device, time_cost, dict2attr
from src.trainer import Trainer


@time_cost("train_profile.txt")
def main():
    # 配置参数
    # config = {
    #     'lr': 1e-3,
    #     'batch_size': 32,
    #     'max_epochs': 3,
    #     'warmup_epochs': 5,
    #     'use_warmup': True,
    #     'device': 'cuda:0',
    #     "seed": 42,
    #     "model_root": "results",
    #     "model_name": "test_transformer",
    #     "current_time": "",
    #     "validloss_interval":10,
    #     "saveloss_interval":10,
    #     "saveckpt_interval":100,

    # }

    config = default_parser().parse_args()
    # config = dict2attr(config)
    config.en_vocab_size = 70608
    config.zh_vocab_size =  5350
    config.d_model = 512
    config.d_ff = 1024
    config.max_seq_length = 5000
    config.dropout = 0.1
    config.nums_heads = 8
    config.num_layers = 6

    config.train_data_path = "./dataset/translation2019zh_train50k.json"
    config.valid_ratio = 0.1
    config.init_lr = 1e-4

    # from src.data.dataloader import get_train_val_loader
    # train_loader, val_loader, en_vocab, zh_vocab, special_tokens = get_train_val_loader(config.train_data_path, batch_size=config.batch_size, val_split=config.valid_ratio)
    config.src_pad_idx = 1         # 1 1 2 3
    config.trg_pad_idx = 1
    config.trg_bos_idx = 2
    config.trg_eos_idx = 3


    import os 
    model_dir = os.path.join(config.model_root, config.model_name)
    os.makedirs(model_dir, exist_ok=True)

    setup_device(config)
    setup_current_time(config)
    logger = initial_distributed_logger(config) ## steup logfile path
    logger.logging_args(config)  ## print hyper-parameters in logfile
    trainer = Trainer(config, logger)
    trainer.initialize(config)
    trainer.training_entrance(config)


if __name__ == '__main__':
    main()
    ## torchrun --nprocnode 2 train.py
    ## python train.py