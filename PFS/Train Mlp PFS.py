import sys
sys.path.append('./main_camel/')

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from camel.dataload.DataLoad_inference import data_load_feature_mlp
from model import Regressor

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)

# Output paths
RESULT_DIR = './results/'
MODEL_DIR = './models/'
DATA_DIR = './data/'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def plot_epoch_diff(diff_list, label):
    epochs = range(len(diff_list))
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, diff_list, color="blue", label=f"{label}_TimeL1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(f"{label}_TimeL1")
    fig.legend()
    plt.savefig(os.path.join(RESULT_DIR, f"{label}_TimeL1.jpg"))
    plt.close()


def main(model, train_path, test_path, batch_size, lr=1e-3, half=False, num_work=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    train_files = os.listdir(train_path)
    train_features, train_labels = [], []
    print('Loading training data...')
    for file in tqdm(train_files, ncols=70):
        data = torch.load(os.path.join(train_path, file), map_location='cpu').view(-1, 1143, 1, 1)
        label = float(file.split('_')[0])
        for sample in data:
            train_features.append(sample)
            train_labels.append([label])

    train_dataset = data_load_feature_mlp(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_work,
                              pin_memory=True, drop_last=False)

    test_files = os.listdir(test_path)
    test_features, test_labels = [], []
    print('Loading testing data...')
    for file in tqdm(test_files, ncols=70):
        data = torch.load(os.path.join(test_path, file), map_location='cpu').view(-1, 1143, 1, 1)
        label = float(file.split('_')[0])
        for sample in data:
            test_features.append(sample)
            test_labels.append([label])

    test_dataset = data_load_feature_mlp(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_work,
                             pin_memory=True, drop_last=False)

    loss_fn = nn.MSELoss().cuda()
    if half:
        model = model.half()

    best_loss = float('inf')
    train_diffs, test_diffs = [], []

    for epoch in range(100):
        model.train()
        epoch_train_diff = []
        for data in tqdm(train_loader, ncols=70):
            inputs, labels = data[0].cuda(), data[1].cuda().float()
            if half:
                inputs, labels = inputs.half(), labels.half()
            inputs[:, :1024, 0, 0] = 0
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_diff.extend((outputs - labels).detach().cpu().numpy())

        avg_train_diff = np.mean(np.abs(epoch_train_diff))
        train_diffs.append(avg_train_diff)

        model.eval()
        epoch_test_diff = []
        for data in tqdm(test_loader, ncols=70):
            inputs, labels = data[0].cuda(), data[1].cuda().float()
            if half:
                inputs, labels = inputs.half(), labels.half()
            inputs[:, :1024, 0, 0] = 0
            outputs = model(inputs)
            epoch_test_diff.extend((outputs - labels).detach().cpu().numpy())

        avg_test_diff = np.mean(np.abs(epoch_test_diff))
        test_diffs.append(avg_test_diff)

        print(f"Epoch {epoch}: Train Diff = {avg_train_diff:.4f}, Test Diff = {avg_test_diff:.4f}")
        plot_epoch_diff(train_diffs, 'train')
        plot_epoch_diff(test_diffs, 'test')

        if avg_test_diff < best_loss:
            best_loss = avg_test_diff
            print(f"Best test loss updated: {best_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_epoch_{epoch}.pt"))
            with open(os.path.join(RESULT_DIR, "train_log.txt"), 'a') as f:
                f.write(f"Epoch {epoch}: Train Diff = {avg_train_diff}, Test Diff = {avg_test_diff}\n")


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.manual_seed(3047)

    model = Regressor(input_size=1143).cuda()
    train_dir = './data/train_features/'
    test_dir = './data/test_features/'
    main(model, train_dir, test_dir, batch_size=1024, lr=1e-5, half=False, num_work=8)
