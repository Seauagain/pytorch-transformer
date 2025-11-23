"""
@author : seauagain
@date : 2025.11.01 
"""


import os 
import time 
import random 
import torch 
import numpy as np 
import platform 
import logging

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
    if not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        args.total_available_gpus = torch.cuda.device_count()
        args.gpu_device_name = torch.cuda.get_device_name(0)
        args.cuda_version = torch.version.cuda
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        args.gpu_memory = f"{total_mem:.2f} GB"
    

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
        if len(args.cuda_ids) >= 2:
            args.use_ddp = True
            args.world_size = len(args.cuda_ids.split(","))
        else:
            args.use_ddp = False
            args.device = f'cuda:{args.cuda_ids[0]}'



def setup_current_time(args):
    r"""Set current time for `.log` file w.r.t differrent platforms."""
    if platform.system()=='Windows':
        args.current_time = time.strftime("%Y-%m-%d %H-%M-%S")
    else:
        args.current_time = time.strftime("%Y-%m-%d %H:%M:%S")



def setup_logging(args):
    r"""Initialize the settings for logging. The `.log` file will be saved in `Model/your_model/log/`."""
    # model_path=os.path.join(args.savedmodel_path,args.current_time)
    # os.makedirs(model_path, exist_ok=True)
    ##move model.py to targetPath
    # shutil.copy('modelclass.py', args.model_path+f'/log/model_{args.current_time}.py')
    # shutil.copy('trainModelClass.py', args.model_path+f'/log/train_{args.current_time}.py')
    # shutil.copy('config.py', args.model_path+f'/log/config_{args.current_time}.py')
    log_path = os.path.join(args.model_root, args.model_name, f'log_{args.current_time}.log')

    # logger = logging.getLogger()
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


def logging_args(args):
    r"""Show hyper-parameters in the header of log file."""
    for key, value in args.__dict__.items():
        logging.info(f"{key}: {value}")
