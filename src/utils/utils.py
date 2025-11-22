"""
@author : seauagain
@date : 2025.11.01 
"""


import os 
import time 
import random 
import torch 
import numpy as np 


def set_seed(seed=42):
    """set random seed for reproduction"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # 以下两行确保cuDNN的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def find_free_port():
    """find a port available for DDP"""
    import socket 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
