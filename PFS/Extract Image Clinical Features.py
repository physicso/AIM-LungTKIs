import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import warnings
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from camel.model.Vit import UNI
from camel.dataload.DataLoad_inference import data_load_inference
from camel.utils import slice_image, load_clinical_features

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)

FEATURE_DIR = './features/'
CLINICAL_CSV_PATH = './data/clinical_features.xlsx'
DATA_DIR = './patches/'
os.makedirs(FEATURE_DIR, exist_ok=True)


def extract_features(model, save_path, data_dir, csv_path, batch_size, half, num_workers, onehot_index,
                     columns, label_col, pathology_col):
    model.eval()
    if half:
        model = model.half()

    clinical_features_list, patient_ids = load_clinical_features(csv_path, onehot_index, columns, label_col, pathology_col)
    patient_ids = [str(pid) for pid in patient_ids]
    files = os.listdir(data_dir)

    for i, file in enumerate(files):
        print(f'{i + 1}/{len(files)}: Processing {file}')
        patch_paths = [os.path.join(data_dir, file, img) for img in os.listdir(os.path.join(data_dir, file))]

        patient_id = file.split('-')[0].lstrip('0') if '-' in file else file.lstrip('0')
        if patient_id not in patient_ids:
            print(f'Skipping {file}: No clinical feature found')
            continue

        index = patient_ids.index(patient_id)
        clinical_vector = torch.tensor(clinical_features_list[index])
        label_df = pd.read_excel(csv_path)
        label = label_df.iloc[index, 5]

        dataset = data_load_inference(patch_paths, file)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

        sm_list, feat_list, fused_list = [], [], []
        for data in tqdm(loader, ncols=70, leave=False):
            inputs = data[0].cuda().half() if half else data[0].cuda()
            with torch.no_grad():
                inputs = slice_image(inputs)
                out, features = model(inputs)
                sm = torch.softmax(out, dim=-1)
                clinical_rep = torch.ones((inputs.size(0), clinical_vector.size(0))) * clinical_vector

            sm_list.append(sm[:, 1].detach().cpu())
            feat_list.append(features[4].detach().cpu())
            fused_list.append(torch.cat([features[4].view(inputs.size(0), -1).detach().cpu(), clinical_rep], dim=-1))

        torch.save(torch.cat(sm_list), os.path.join(save_path, 'sm_camel', f'{label}_{file}.pt'))
        torch.save(torch.cat(feat_list), os.path.join(save_path, 'feature', f'{label}_{file}.pt'))
        torch.save(torch.cat(fused_list), os.path.join(save_path, 'feature_clinical', f'{label}_{file}.pt'))


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

    model = UNI(Linear_only=False).cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    for phase in ['train', 'test']:
        for polarity in ['pos_image', 'neg_image']:
            data_dir = f'./patches/{phase}/{polarity}/'
            save_dir = f'./features/{phase}/'
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'feature'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'feature_clinical'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'sm_camel'), exist_ok=True)

            extract_features(
                model=model,
                save_path=save_dir,
                data_dir=data_dir,
                csv_path=CLINICAL_CSV_PATH,
                batch_size=6,
                half=False,
                num_workers=8,
                onehot_index=203,
                columns=[6, 96],
                label_col=5,
                pathology_col=2
            )
