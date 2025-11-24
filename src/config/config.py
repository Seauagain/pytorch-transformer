import argparse

def default_parser():

    parser = argparse.ArgumentParser(description='Torch Training')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # ========================= General Configs ===========================
    # parser.add_argument('--mech_path', type=str, help='chemical mechanism file dir')
    # parser.add_argument('--zero_input', nargs='+', default=["Ar", "He"], type=str, help='species with zero values in the input dataset.')
    # parser.add_argument('--zero_gradient', nargs='+', default=[ ], type=str, help='species with zero rate of change')

    # ========================= DataLoader Configs ==========================
    parser.add_argument('--train_data_path', type=str, help='the path of training dataset')
    parser.add_argument('--shuffle', action='store_false', help='shuffle the training dataset')      # default true
    parser.add_argument('--batch_size', default=32, type=int, help='use for training duration per worker')
    parser.add_argument('--valid_batch_size', default=32, type=int, help="use for validation duration per worker")
    parser.add_argument('--train_size', type=int, help='training dataset size')
    parser.add_argument('--valid_size', type=int, help='validation dataset size')
    parser.add_argument('--valid_interval', default=10, type=int, help='validation interval in epochs')
    parser.add_argument('--valid_ratio', default=0.1, type=float, help='split percentages of training data as validation')
    parser.add_argument('--prefetch', default=10, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    parser.add_argument('--pin_memory', action='store_false', help="pin_memory for DataLoader")     #default True

    # ========================= Network Configs =====================
    parser.add_argument('--dim', default=-1, type=int, help='feature dimention of data')
    parser.add_argument('--input_dim', default=-1, type=int, help='input dimention of data')
    parser.add_argument('--label_dim', default=-1, type=int, help='label dimention of data')
    parser.add_argument('-l', '--layers', nargs='+', default=[1600, 800, 400], type=int, help='dnn hidden layers')
    parser.add_argument('--net_type', default='fc', type=str, help='dnn type')
    parser.add_argument('--actfun', default='gelu', type=str, help='activation function')

    # ========================= Training Configs =======================
    parser.add_argument('--model_root', default="results", type=str, help='the default root dir to store models')
    parser.add_argument('--model_name', default="test_model", type=str, help='the default model name')
    parser.add_argument('--max_epochs', default=3, type=int, help='max epochs in training')
    parser.add_argument('--warmup_epochs', default=1, type=int, help='the number of warm up epochs')
    parser.add_argument('--validloss_interval', default=10, type=int)
    parser.add_argument('--saveloss_interval', default=10, type=int)
    parser.add_argument('--saveckpt_interval', default=100, type=int)

    parser.add_argument('--lossfun', default='L1', type=str, help='loss funtion: MSE,L1/MAE,CEl')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer for training')
    parser.add_argument('--device', default='cuda:0', type=str, help='default device')

    # ========================= Multi-GPU Configs =======================
    parser.add_argument('--total_available_gpus', default=0, type=int, help='gpu numbers of the machine')
    parser.add_argument('--gpu_device_name', default="", type=str, help='gpu type')
    parser.add_argument('--gpu_memory', default=0, type=float, help='gpu memory')
    parser.add_argument('--cuda_version', default="", type=str, help='cuda version')
    parser.add_argument('-ddp', '--use_ddp', action='store_true', help='whether use DistributedDataParallel') #default=False
    # DDP CUDA available device
    parser.add_argument('-cudas', '--cuda_ids', default="-1", type=str, help='no use') #default=False
    parser.add_argument('--backend', default='nccl', type=str, help='current process backend for DDP, gloo,nccl,mpi')

    # ========================= Other Configs =======================
    parser.add_argument('--current_time', default='today', type=str, help='record the start time')
    parser.add_argument('-note', '--description', default='test', type=str, help='description of the experiment(purpose/target/motivation)')

    return parser
