"""
@author : seauagain
@date : 2025.11.01 
"""


import os, sys
import time 
import random 
import torch 
import numpy as np 
import platform 
import logging
from typing import Any


class dict2attr(dict):
    """dict['key'] to dict.key"""
    def __getattr__(self, item):
        return self[item] 

def set_seed(args):
    """set random seed for reproduction"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
        # 以下两行确保cuDNN的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_device(args):
    r"""Set the device for training and the default device is `cuda:0`. 
        If torch.cuda is not available, use cpu instead.
    """
    if torch.cuda.is_available():
        args.total_available_gpus = torch.cuda.device_count()
        args.gpu_device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        args.gpu_memory = f"{total_mem:.2f} GB"
        args.cuda_version = torch.version.cuda

        ddp_env = (
                torch.distributed.is_available()
                and "WORLD_SIZE" in os.environ
                and int(os.environ['WORLD_SIZE']) >=2
                )
        if ddp_env: #DDP
            args.use_ddp = True
            args.world_size = int(os.environ['WORLD_SIZE'])
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                cuda_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
            else:
                cuda_ids = [i for i in range( int(os.environ['WORLD_SIZE']) )]
            args.device = f'cuda={cuda_ids}'
        else: # single GPU
            args.use_ddp = False
    else:
        args.device = 'cpu'

def setup_current_time(args):
    r"""Set current time for `.log` file w.r.t differrent platforms."""
    if platform.system()=='Windows':
        args.current_time = time.strftime("%Y-%m-%d %H-%M-%S")
    else:
        args.current_time = time.strftime("%Y-%m-%d %H:%M:%S")

class DistributedLogger:
    def __init__(self, log_path="run.log"):
        self.is_main_proc = self._check_is_main_process()
        self.setup_logging(log_path)
    
    def _check_is_main_process(self) -> bool:
        """chenck the current process wheher main process."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        elif 'RANK' in os.environ:
            return int(os.environ['RANK']) == 0
        else:
            return True  # non-distributed env

    def setup_logging(self, log_path="run.log"):
        r"""Initialize the settings for logging."""
        # log_path = os.path.join(args.model_root, args.model_name, f'log_{args.current_time}.log')
        logger=logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s][%(name)s]:%(message)s')

        # clear handlers to avoid duplicate log
        if (logger.hasHandlers()):
            logger.handlers.clear()

        # create streamhandler for terminal
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # create filehandler for .log file
        sh = logging.FileHandler(str(log_path))
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # add new handler
        # logger.propagate=False
        return logger
    
    def info(self, msg: str, *arg: Any, process: str = "main"):
        if process == "all" or self.is_main_proc:
            logging.info(msg, *arg)
        
    def logging_args(self, args):
        r"""Show hyper-parameters in the header of log file."""
        if self.is_main_proc:
            for key, value in args.__dict__.items():
                logging.info(f"{key}: {value}")
                
            import shlex
            logging.info(f'command: {shlex.join(sys.argv)}')
            


def initial_distributed_logger(args):
    log_path = os.path.join(args.model_root, args.model_name, f'log_{args.current_time}.log')
    logger = DistributedLogger(log_path)
    return logger 


def time_cost(output_file="profile_stats.txt"):
    import cProfile
    import pstats
    import io
    from functools import wraps
    """
    Use cProfile to profile the function's execution time and output the results to a file.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()  # start

            result = func(*args, **kwargs)

            pr.disable() # end
            
            # save
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
            ps.print_stats()
            with open(output_file, "w") as f:
                f.write(s.getvalue())
            print(f"[cProfile] Profile saved to: {output_file}")
            return result
        return wrapper
    return decorator





'''deprecated code
if ddp_env: #DDP
    args.use_ddp = True
    user_cuda_ids = args.cuda_ids.split(",")
    if len(user_cuda_ids) >= int(os.environ['WORLD_SIZE']): # user-specific cuda ids
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.device = f'cuda={user_cuda_ids[:args.world_size]}'
    else:
        user_cuda_ids = [i for i in range( int(os.environ['WORLD_SIZE']) )]
        args.device = f'cuda={user_cuda_ids}'
'''