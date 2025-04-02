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
import torchvision
from tqdm import tqdm
import cv2 as cv
from camel.model.resnet import ResNet18
from camel.model.swin_transformer_v2 import Swinv2_L
from camel.train.camel_frame import train

from camel.distributed import get_ddp_generator
from camel.eval import camel_relabel
from camel.model.camel_feature import Swinv2_L_feature
from camel.dataload.DataLoad_inference import data_load_inference
from camel.utils import slice_image
from camel.model.EfficientNet import Efficientv2_l_pretrained
from camel.model.Vit import VIT_L, VIT_B, VIT_H
from camel.model.regnet import regnet_y_32gf_pretrained, regnet_y_400mf_pretrained
from camel.model.Moe import MoE
from model import all_512_pfs, Dino_Mlp

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)


def main(model, save_path, data_dir, BatchSize, half=False, label=None, num_work=8):
    model.eval()
    if half:
        model = model.half()

    path_dir = os.listdir(data_dir)  # Slice names
    num = len(path_dir)

    # Create save directories if they don't exist
    if os.path.exists(save_path):
        print('Save path already exists')
    else:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}/feature', exist_ok=True)
        os.makedirs(f'{save_path}/sm_camel', exist_ok=True)
        os.makedirs(f'{save_path}/sm_class', exist_ok=True)
        print('Directories created successfully')

    # Determine label automatically if not provided
    if label is None:
        if 'pos' in data_dir:
            label = 1
        elif 'neg' in data_dir:
            label = 0
        else:
            raise ValueError('Label determination failed')

    for i in range(0, num):
        print(f'{i+1}/{num}, {data_dir + path_dir[i]}')

        img_dir = os.listdir(os.path.join(data_dir, path_dir[i]))
        num2 = len(img_dir)  # Number of patches

        image_path = []
        for k in range(0, num2):
            image_path.append(os.path.join(data_dir, path_dir[i], img_dir[k]))  # Patch paths

        name = path_dir[i]  # Slice name

        dataset = data_load_inference(image_path, name)
        inference_data = DataLoader(
            dataset=dataset,
            batch_size=BatchSize,
            shuffle=False,
            num_workers=num_work,
            pin_memory=True,
            drop_last=False
        )

        sm_save = []
        for _, data in enumerate(tqdm(inference_data, 0, leave=False, ncols=70)):
            name = data[1][0]

            if half:
                inputs = data[0].half().cuda()
            else:
                inputs = data[0].cuda()

            with torch.no_grad():
                inputs = slice_image(inputs)
                outputs = model(inputs.squeeze(0))
                sm = torch.softmax(outputs, dim=-1)

            sm_save.append(sm[:, 1].detach().cpu())

        sm_result = torch.cat(sm_save, dim=0)
        os.makedirs(f'{save_path}/sm_camel', exist_ok=True)
        torch.save(sm_result, f'{save_path}/sm_camel/{label}_{name}.pt')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

    # Initialize model and load weights
    model = all_512_pfs().cuda()
    model.load_state_dict(torch.load(
        './checkpoints/37_分类.pt',  # Example path, replace with your actual checkpoint path
        map_location=torch.device('cpu')
    ))

    # Save path (REPLACE WITH YOUR SAVE PATH)
    save_path = './inference_results/'  # Example path, replace with your actual save path

    # Positive data directory (REPLACE WITH YOUR POSITIVE DATA PATH)
    pos_data_dir = './data/train/pos_image/'  # Example path, replace with your actual positive data path
    main(model, save_path, pos_data_dir, BatchSize=2, half=False, label=None, num_work=8)

    # Negative data directory (REPLACE WITH YOUR NEGATIVE DATA PATH)
    neg_data_dir = './data/train/neg_image/'  # Example path, replace with your actual negative data path
    main(model, save_path, neg_data_dir, BatchSize=2, half=False, label=None, num_work=8)