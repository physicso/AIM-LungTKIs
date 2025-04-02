import sys
sys.path.append('/home/yangmoxuan/main_camel/')

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
import numpy as np
from camel.dataload.DataLoad_inference import data_load_feature_sm
from model import Regressor

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)

DATA_DIR = './data/'
OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(model, data_path, save_path, batch_size, half=False, num_work=8):
    file_list = os.listdir(data_path)
    model.eval()
    slice_level_errors = []

    for file_name in tqdm(file_list, ncols=70):
        slice_result = []
        patch_features = torch.load(os.path.join(data_path, file_name), map_location='cpu').view(-1, 1143, 1, 1)
        label = float(file_name.split('_')[0])

        inputs = [patch_features[i] for i in range(patch_features.shape[0])]
        dataset = data_load_feature_sm(inputs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_work,
                            pin_memory=True, drop_last=False)

        patch_errors = []
        for data in tqdm(loader, ncols=70, leave=False):
            In = data.cuda()
            In[:, :1024, 0, 0] = 0
            with torch.no_grad():
                out = model(In.unsqueeze(0))
            patch_errors.extend((out - label).detach().cpu().numpy())
            slice_result.append(out.detach().cpu())

        avg_patch_error = np.mean(np.abs(patch_errors))
        slice_level_errors.append(avg_patch_error)

        sm_result = torch.cat(slice_result, dim=0)
        torch.save(sm_result, os.path.join(save_path, file_name))

    total_avg_error = np.mean(slice_level_errors)
    print(f"Average Slice-Level Error: {total_avg_error:.4f}")


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.manual_seed(3047)

    model = Regressor(input_size=1143).cuda()
    model.load_state_dict(torch.load("./models/best_epoch_12.pt", map_location='cpu'))

    for phase in ['train', 'test']:
        input_path = os.path.join(DATA_DIR, f'{phase}_features')
        output_path = os.path.join(OUTPUT_DIR, f'{phase}_inference')
        os.makedirs(output_path, exist_ok=True)
        main(model, input_path, output_path, batch_size=1024, num_work=8)
