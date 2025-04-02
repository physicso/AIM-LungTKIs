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
import numpy as np
from tqdm import tqdm
import cv2 as cv
import pandas as pd
from camel.model.resnet import ResNet18
from camel.train.camel_frame import train
from camel.model.EfficientNet import Efficientv2_l_pretrained, Efficientv2_m_pretrained
from camel.distributed import get_ddp_generator
from camel.eval import camel_relabel
from camel.dataload.DataLoad_inference import data_load_inference
from camel.utils import slice_image, load_clinical_features
from camel.model.resnet import ResNet50_pretrained

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)


def main(model, save_path, data_dir, csv_path, BatchSize, half=False, label=None, num_work=8, onehot_index=57, columns=[1, 13], label_columns=16, Pathology_columns=0):
    model.eval()
    if half:
        model = model.half()

    path_dir = os.listdir(data_dir)
    num = len(path_dir)

    # Create save directories if they don't exist
    if os.path.exists(save_path):
        print('Save path already exists')
    else:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}/feature', exist_ok=True)
        os.makedirs(f'{save_path}/feature_clinical', exist_ok=True)
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

    clinical_features_list, Pathology_number = load_clinical_features(csv_path, onehot_index, columns, label_columns, Pathology_columns)
    Pathology_number = [str(x) for x in Pathology_number]

    for i in range(0, num):
        print(f'{i+1}/{num}, {data_dir + path_dir[i]}')

        img_dir = os.listdir(os.path.join(data_dir, path_dir[i]))
        num2 = len(img_dir)

        image_path = []
        for k in range(0, num2):
            image_path.append(os.path.join(data_dir, path_dir[i], img_dir[k]))

        name = path_dir[i]  # Slice name
        if '-' in name:
            match_name = str(name).split('-')[0]
            slice_name_no_zero = match_name.lstrip('0')  # Remove leading zeros
        else:
            match_name = str(name)
            slice_name_no_zero = match_name.lstrip('0')

        if slice_name_no_zero not in Pathology_number:
            print(f'Clinical features not found, skipping {name}')
            continue

        index = Pathology_number.index(slice_name_no_zero)  # Match clinical features by name
        clinical_features = clinical_features_list[index]
        clinical_features = torch.tensor(clinical_features)

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
        feature_save = []
        feature_clinical_save = []

        for _, data in enumerate(tqdm(inference_data, 0, leave=False, ncols=70)):
            if half:
                inputs = data[0].half().cuda()
            else:
                inputs = data[0].cuda()

            with torch.no_grad():
                inputs = slice_image(inputs)
                outputs, features = model(inputs)

                sm = torch.softmax(outputs, dim=-1)

                clinical_save = torch.ones(size=(inputs.shape[0], clinical_features.shape[0])) * clinical_features

            sm_save.append(sm[:, 1].detach().cpu())
            feature_save.append(features['feature6'].detach().cpu())
            feature_clinical_save.append(torch.cat([features['feature6'].view(inputs.shape[0], -1).detach().cpu(), clinical_save], dim=-1))

        sm_result = torch.cat(sm_save, dim=0)
        feature_result = torch.cat(feature_save, dim=0)
        feature_clinical_result = torch.cat(feature_clinical_save, dim=0)

        os.makedirs(f'{save_path}/sm_camel', exist_ok=True)
        os.makedirs(f'{save_path}/feature', exist_ok=True)
        os.makedirs(f'{save_path}/feature_clinical', exist_ok=True)

        torch.save(sm_result, f'{save_path}/sm_camel/{label}_{name}.pt')
        torch.save(feature_result, f'{save_path}/feature/{label}_{name}.pt')
        torch.save(feature_clinical_result, f'{save_path}/feature_clinical/{label}_{name}.pt')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

    model = Efficientv2_m_pretrained(Linear_only=False).cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert BN layers
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # Training data directories and save paths (REPLACE WITH YOUR PATHS)
    save_path_train = './train_results/'  # Example path, replace with your actual train save path
    csv_path_train = './data/cli_label_0601_PR_onehot.xlsx'  # Example path, replace with your actual CSV path
    data_dir_pos_train = './data/train/pos_image/'  # Example path, replace with your actual positive train data path
    data_dir_neg_train = './data/train/neg_image/'  # Example path, replace with your actual negative train data path

    main(model, save_path_train, data_dir_pos_train, csv_path_train, BatchSize=1, half=False, label=None, num_work=8)
    main(model, save_path_train, data_dir_neg_train, csv_path_train, BatchSize=1, half=False, label=None, num_work=8)

    # Testing data directories and save paths (REPLACE WITH YOUR PATHS)
    save_path_test = './test_results/'  # Example path, replace with your actual test save path
    csv_path_test = './data/cli_label_0601_PR_onehot.xlsx'  # Example path, replace with your actual CSV path
    data_dir_pos_test = './data/test/pos_image/'  # Example path, replace with your actual positive test data path
    data_dir_neg_test = './data/test/neg_image/'  # Example path, replace with your actual negative test data path

    main(model, save_path_test, data_dir_pos_test, csv_path_test, BatchSize=1, half=False, label=None, num_work=8)
    main(model, save_path_test, data_dir_neg_test, csv_path_test, BatchSize=1, half=False, label=None, num_work=8)