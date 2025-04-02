import sys
sys.path.append('./main_camel/')  # Add project path to system path

import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import warnings
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2 as cv
from camel.model.resnet import ResNet18
from camel.model.swin_transformer_v2 import Swinv2_L
from camel.dataload.DataLoad_inference import data_load_feature_sm
from camel.utils import slice_image
from model import all_512_pfs, Dino_Mlp

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)


def main(model, data_path, save_path, BatchSize, half=False, label=None, num_work=8):
    pt_path = os.listdir(data_path)
    num = len(pt_path)

    model.eval()

    for i in tqdm(range(0, num), 0, leave=False, ncols=70):
        sm_save = []
        camel_result = []
        camel_sm = torch.load(f'{data_path}{pt_path[i]}', map_location=torch.device('cpu')).view(-1, 2172, 1, 1)

        for kk in range(0, camel_sm.shape[0]):
            camel_result.append(camel_sm[kk])

        dataset = data_load_feature_sm(camel_result)
        inference_data = DataLoader(
            dataset=dataset,
            batch_size=BatchSize,
            shuffle=False,
            num_workers=num_work,
            pin_memory=True,
            drop_last=False
        )

        for k, data in enumerate(tqdm(inference_data, 0, leave=False, ncols=70)):
            inputs = data[0].cuda()
            outputs = model(inputs.unsqueeze(0))
            sm = torch.softmax(outputs, dim=-1)
            sm_save.append(sm[:, 1].detach().cpu())

        sm_result = torch.cat(sm_save, dim=0)
        os.makedirs(save_path, exist_ok=True)
        torch.save(sm_result, f'{save_path}{pt_path[i]}')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.random.manual_seed(3047)

    model = Dino_Mlp().cuda()
    model.load_state_dict(torch.load(
        './checkpoints/42_分类.pt',  # Example path, replace with your actual checkpoint path
        map_location=torch.device('cpu')
    ))

    # Test data path and save path (REPLACE WITH YOUR TEST DATA AND SAVE PATHS)
    test_data_path = './data/test_features/'  # Example path, replace with your actual test data path
    test_save_path = './inference_results/test/'  # Example path, replace with your actual test save path
    main(model, test_data_path, test_save_path, BatchSize=32, num_work=8)

    # Train data path and save path (REPLACE WITH YOUR TRAIN DATA AND SAVE PATHS)
    train_data_path = './data/train_features/'  # Example path, replace with your actual train data path
    train_save_path = './inference_results/train/'  # Example path, replace with your actual train save path
    main(model, train_data_path, train_save_path, BatchSize=32, num_work=8)