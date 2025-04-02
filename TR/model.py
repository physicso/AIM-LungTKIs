import matplotlib.pyplot as plt
import datetime
import numpy as np
import math
import tqdm
from tqdm import tqdm
import math
import sys
import warnings
import pandas as pd 
import random
import os
import sys
import warnings
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torch.optim import lr_scheduler
import torch.functional as f
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class all_512_pfs(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2172, 2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out=self.linear1(x)
        return out

class Dino_Mlp(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2172, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128,2)
        
    def forward(self, x):# iuput B*45   output B*2
        out=x.view(x.size(0), -1)
        out=self.fc1(out)
        out=self.relu(out)
        
        out=self.fc2(out)
        
        return out
