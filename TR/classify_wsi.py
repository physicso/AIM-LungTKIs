import sys
sys.path.append('./main_camel/')  # Add project path to system path

import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import warnings
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from camel.model.Vit import VIT_L, VIT_B
from camel.train.classify_frame import train, test, train_with_test
from camel.dataload.DataLoad_WSI import data_load_classifiy
from camel.distributed import get_ddp_generator
from camel.model.convnextv2 import convnextv2_L_pretrained, convnextv2_H_pretrained, convnextv2_T_pretrained
from camel.main.main_classify_WSI import main
from camel.model.Moe import SoftMoELayerWrapper, PEER, DeepseekV2MoE

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)


class DeepseekV2MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_moe = DeepseekV2MoE(
            num_expert=8,
            top_k=2,
            num_share_expert=0,
            input_dim=1536,
            output_dim=64,
            keep_shape=False,
            experts_layer=DeepseekV2MLP(
                in_features=1536,
                hidden_features=128,
                out_features=64,
                act_layer=nn.GELU,
                drop=0.1
            ),
            share_experts_layer=None,
            batch_gate=False,
            feature_length=None
        ).cuda()

        self.bn = nn.BatchNorm2d(1536, momentum=0.9)
        self.avp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)
        self.drop = nn.Dropout2d(p=0.1)

    def forward(self, x):
        x = self.drop(x)
        x = self.mlp_moe(x)
        x = self.avp(x)
        x = x.view(1, -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('-l', '--learn_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('-e', '--epoch', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-seed', '--seed', type=int, default=3047, help='Random seed')
    parser.add_argument('-amp', '--amp', type=bool, default=False, help='Mixed precision')
    parser.add_argument('-pt', '--pt', type=bool, default=False, help='Load weights')
    parser.add_argument('-ws', '--world_size', type=int, default=1, help='Number of GPUs')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.random.manual_seed(args.seed)

    model_name = 'only_image'

    # Data paths (REPLACE WITH YOUR DATA PATHS)
    data_paths = {
        'train_pos_dir': './data/train_pos.txt',  # Example path, replace with your actual path
        'train_neg_dir': './data/train_neg.txt',  # Example path, replace with your actual path
        'test_pos_dir': './data/test_pos.txt',    # Example path, replace with your actual path
        'test_neg_dir': './data/test_neg.txt'     # Example path, replace with your actual path
    }

    # Save paths (REPLACE WITH YOUR SAVE PATHS)
    save_paths = {
        'pt_path': './checkpoints/',              # Example path, replace with your actual path
        'logging_path': './logs/'                 # Example path, replace with your actual path
    }

    model = Net().cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert BN layers
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    grad_freeze = None

    main(model, args, model_name, data_paths, save_paths, mode='normal', optimizer='Adam', grad_freeze=grad_freeze)