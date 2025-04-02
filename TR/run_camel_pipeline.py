import sys
sys.path.append('./main_camel/')  # Add project path to system path

import torch
import argparse
import os
import torch.distributed as dist
import warnings
from camel.model.swin_transformer_v2 import Swinv2_L
from camel.model.EfficientNet import Efficientv2_m_pretrained
from camel.main.main_camel_DDP import main
from camel.model.regnet import regnet_y_400mf_pretrained
from camel.model.Vit import UNI
from camel.model.resnet import ResNet50_pretrained

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)

#### Training PR dataset with Camel, binary classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('-l', '--learn_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-seed', '--seed', type=int, default=3047, help='Random seed')
    parser.add_argument('-amp', '--amp', type=bool, default=False, help='Mixed precision')
    parser.add_argument('-pt', '--pt', type=bool, default=False, help='Load weights')
    parser.add_argument('-ws', '--world_size', type=int, default=4, help='Number of GPUs')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.random.manual_seed(args.seed)

    model_name = 'ResNet50_pretrained_Camel'

    # Data paths (REPLACE WITH YOUR DATA PATHS)
    data_paths = {
        'train_pos_dir': './data/train_pos_txt.txt',  # Example path, replace with your actual path
        'train_neg_dir': './data/train_neg_txt.txt'   # Example path, replace with your actual path
    }

    # Save paths (REPLACE WITH YOUR SAVE PATHS)
    save_paths = {
        'pt_path': './checkpoints/',                  # Example path, replace with your actual path
        'logging_path': './logs/'                     # Example path, replace with your actual path
    }

    # Initialize model
    model = ResNet50_pretrained(Linear_only=True).cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert BN layers
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # Start training
    main(model, args, model_name, data_paths, save_paths, optimizer='Adam', grad_freeze=None)