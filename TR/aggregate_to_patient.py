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
from camel.model.swin_transformer_v2 import Swinv2_L
from camel.train.camel_frame import train
from camel.distributed import get_ddp_generator
from camel.eval import camel_relabel
from camel.dataload.DataLoad_inference import data_load_feature_mlp
from camel.utils import slice_image
from camel.utils import roc
from sklearn.metrics import roc_auc_score

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)

# Load Excel file (REPLACE WITH YOUR EXCEL PATH)
df = pd.read_excel('./data/cli_label_0601_PR_onehot.xlsx')  # Example path, replace with your actual path
patient_ids = df['patient IDs']  # Column name for patient IDs

# Directory paths (REPLACE WITH YOUR DIRECTORY PATHS)
input_dir = './input_data/'  # Example path, replace with your actual input directory
output_dir = './output_data/'  # Example path, replace with your actual output directory

path_dir = os.listdir(input_dir)
num = len(path_dir)

# Create output directory if it doesn't exist
if os.path.exists(output_dir):
    print('Directory exists')
else:
    os.makedirs(output_dir, exist_ok=True)

patient_slice_data = {}

for i in range(num):
    # Process file name to extract slice name and label
    file_name = path_dir[i]
    label = file_name.split('_')[0]
    slice_name = file_name.split('_')[1]
    
    if '-' in slice_name:
        slice_name_part = slice_name.split('-')[0]
        slice_name_no_zero = slice_name_part.lstrip('0')  # Remove leading zeros
    else:
        slice_name_part = slice_name
        slice_name_no_zero = slice_name_part.lstrip('0')
        slice_name_no_zero = slice_name_no_zero.split('.')[0]
    
    # Load data
    sm_class = torch.load(os.path.join(input_dir, file_name), map_location=torch.device('cpu'))
    
    # Organize data by patient ID
    if slice_name_no_zero not in patient_slice_data:
        patient_slice_data[slice_name_no_zero] = [(label, sm_class)]
    else:
        patient_slice_data[slice_name_no_zero].append((label, sm_class))

# Save combined data for each patient
for patient_id, label_sm_class_list in patient_slice_data.items():
    sm_class_list = [sm_class for label, sm_class in label_sm_class_list]
    labels = [label for label, sm_class in label_sm_class_list]
    
    if len(sm_class_list) > 1:
        sm_class_combined = torch.cat(sm_class_list, dim=0)
    else:
        sm_class_combined = sm_class_list[0]
    
    label = labels[0]
    output_file_path = os.path.join(output_dir, f'{label}_{patient_id}.pt')
    torch.save(sm_class_combined, output_file_path)