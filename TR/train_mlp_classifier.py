import sys
sys.path.append('./main_camel/')  # Add project path to system path

import random
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
from camel.model.resnet import ResNet18
from camel.model.swin_transformer_v2 import Swinv2_L
from camel.train.camel_frame import train
from camel.distributed import get_ddp_generator
from camel.eval import camel_relabel
from camel.dataload.DataLoad_inference import data_load_feature_mlp
from camel.utils import slice_image
from camel.utils import roc
from model import all_512_pfs, Dino_Mlp
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)


class FocalLoss(nn.Module):
    def __init__(self, gamma=3, alpha=0.2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss * self.alpha)


def roc(y_true, y_score, sklearn=True):
    pos_label = 1
    num_positive_examples = (y_true == pos_label).sum()
    num_negative_examples = len(y_true) - num_positive_examples

    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score) + 1

    for i in np.flip(np.argsort(y_score)):
        if y_score[i] != score:
            fpr.append(fp / num_negative_examples)
            tpr.append(tp / num_positive_examples)
            thresholds.append(score)
            score = y_score[i]

        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / num_negative_examples)
    tpr.append(tp / num_positive_examples)
    thresholds.append(score)

    maxindex = (np.array(tpr) - np.array(fpr)).tolist().index(max(np.array(tpr) - np.array(fpr)))
    cutoff = thresholds[maxindex]
    index = thresholds.index(cutoff)

    se = tpr[index]
    sp = 1 - fpr[index]

    if sklearn:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(len(tpr) - 1):
            auc += (fpr[i + 1] - fpr[i]) * tpr[i + 1]

    correct = 0
    for i in range(y_score.shape[0]):
        if y_score[i] >= cutoff and y_true[i] == 1:
            correct += 1
        elif y_score[i] < cutoff and y_true[i] == 0:
            correct += 1
    acc = correct / y_score.shape[0] * 100

    return auc, se, sp, index, fpr, tpr, cutoff, acc


def auc_figure(auc, se, sp, index, fpr, tpr, cutoff, acc, type):

    fig, ax = plt.subplots()
    plt.plot([0, 1], '--')
    plt.plot(fpr[index], tpr[index], 'bo')
    ax.text(fpr[index], tpr[index] + 0.02, f'cut_off={round(cutoff, 3)}', fontdict={'fontsize': 10})
    plt.plot(fpr, tpr)
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    text = f'AUC:{round(auc, 3)}\nSE:{round(se, 3)}\nSP:{round(sp, 3)}\nAccuracy:{round(acc, 3)}%\n'
    ax.text(0.6, 0.05, text, fontsize=12)

    plt.savefig(f'./results/{type}.png')  # Example path, replace with your actual path


def main(model, train_path, test_path, BatchSize, lr=1e-3, half=False, label=None, num_work=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    # Load training data
    pt_path = os.listdir(train_path)
    num = len(pt_path)

    true_labels = []
    features = []

    print('Loading training data')
    for i in tqdm(range(0, num), 0, leave=False, ncols=70):
        feature = torch.load(f'{train_path}{pt_path[i]}', map_location=torch.device('cpu')).view(-1, 2172, 1, 1)
        for kk in range(0, feature.shape[0]):
            features.append(feature[kk])
            label = int(pt_path[i][0])
            true_labels.append(label)

    dataset = data_load_feature_mlp(features, true_labels)
    inference_data = DataLoader(
        dataset=dataset,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=num_work,
        pin_memory=True,
        drop_last=False
    )

    # Load testing data
    pt_path = os.listdir(test_path)
    num = len(pt_path)

    true_labels = []
    features = []

    print('Loading testing data')
    for i in tqdm(range(0, num), 0, leave=False, ncols=70):
        feature = torch.load(f'{test_path}{pt_path[i]}', map_location=torch.device('cpu')).view(-1, 2172, 1, 1)
        for kk in range(0, feature.shape[0]):
            features.append(feature[kk])
            label = int(pt_path[i][0])
            true_labels.append(label)

    dataset = data_load_feature_mlp(features, true_labels)
    inference_data_test = DataLoader(
        dataset=dataset,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=num_work,
        pin_memory=True,
        drop_last=False
    )

    ce_loss = nn.CrossEntropyLoss().cuda()

    print('Starting training')

    if half:
        model = model.half()
    best_auc = 0
    for epoch in range(0, 100):
        model.train()

        for i, data in enumerate(tqdm(inference_data, 0, leave=False, ncols=70)):
            if half:
                inputs = data[0].half().cuda()
                labels = data[1].half().cuda()
            else:
                inputs = data[0].cuda()
                labels = data[1].cuda()
            inputs[:, 2048:, 0, 0] = 0
            optimizer.zero_grad()

            outputs = model(inputs.squeeze(0))
            sm = torch.softmax(outputs, dim=-1)

            loss = ce_loss(outputs, labels.long())

            loss.backward()
            optimizer.step()

            pred_labels = []
            pred_1 = []
            for lb in range(labels.shape[0]):
                pred_labels.append(labels[lb].item())
                pred_1.append(sm[lb, 1].item())
            pred_labels = torch.tensor(pred_labels).cuda()
            pred_1 = torch.tensor(pred_1).cuda()

            if i == 0:
                pred_roc = pred_1.detach().cpu().numpy()
                label_roc = pred_labels.detach().cpu().numpy()
            else:
                pred_roc = np.concatenate((pred_roc, pred_1.detach().cpu().numpy().reshape(-1)), axis=0)
                label_roc = np.concatenate((label_roc, pred_labels.detach().cpu().numpy().reshape(-1)), axis=0)

        model.eval()

        for i, data in enumerate(tqdm(inference_data_test, 0, leave=False, ncols=70)):
            if half:
                inputs = data[0].half().cuda()
                labels = data[1].half().cuda()
            else:
                inputs = data[0].cuda()
                labels = data[1].cuda()
            inputs[:, 2048:, 0, 0] = 0
            outputs = model(inputs.squeeze(0))

            sm = torch.softmax(outputs, dim=-1)

            pred_labels = []
            pred_1 = []
            for lb in range(labels.shape[0]):
                pred_labels.append(labels[lb].item())
                pred_1.append(sm[lb, 1].item())
            pred_labels = torch.tensor(pred_labels).cuda()
            pred_1 = torch.tensor(pred_1).cuda()

            if i == 0:
                pred_roc_test = pred_1.detach().cpu().numpy()
                label_roc_test = pred_labels.detach().cpu().numpy()
            else:
                pred_roc_test = np.concatenate((pred_roc_test, pred_1.detach().cpu().numpy().reshape(-1)), axis=0)
                label_roc_test = np.concatenate((label_roc_test, pred_labels.detach().cpu().numpy().reshape(-1)), axis=0)

        auc_train, se_train, sp_train, index_train, fpr_train, tpr_train, cutoff_train, acc_train = roc(label_roc, pred_roc)
        auc_test, se_test, sp_test, index_test, fpr_test, tpr_test, cutoff_test, acc_test = roc(label_roc_test, pred_roc_test)

        print(f'Train AUC: {auc_train}, Test AUC: {auc_test}')

        if auc_test > best_auc:
            best_auc = auc_test
            print(f'>>>>>>>>>>>>>>>>>>>Test AUC: {auc_test}<<<<<<<<<<<<<<<<<<<<<<<<')

            auc_figure(auc_train, se_train, sp_train, index_train, fpr_train, tpr_train, cutoff_train, acc_train, 'train')
            auc_figure(auc_test, se_test, sp_test, index_test, fpr_test, tpr_test, cutoff_test, acc_test, 'test')
            os.makedirs('./checkpoints/', exist_ok=True)
            torch.save(model.state_dict(), f'./checkpoints/{epoch}_分类.pt')  # Example path, replace with your actual path


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.random.manual_seed(3047)

    model = Dino_Mlp().cuda()

    # Training data path (REPLACE WITH YOUR TRAINING DATA PATH)
    train_path = './data/train_features/'  # Example path, replace with your actual training data path

    # Testing data path (REPLACE WITH YOUR TESTING DATA PATH)
    test_path = './data/test_features/'    # Example path, replace with your actual testing data path

    main(model, train_path=train_path, test_path=test_path, BatchSize=1024, lr=3e-4, half=False, label=None, num_work=8)