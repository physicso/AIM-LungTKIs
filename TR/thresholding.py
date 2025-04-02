import sys
sys.path.append('./main_camel/')  # Add project path to system path

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from camel.eval import range_threshold

# Data path (REPLACE WITH YOUR DATA PATH)
data_path = './data/'  # Example path, replace with your actual data path
pt_path = os.listdir(data_path)
num = len(pt_path)

auc_best = 0
cutoff = 0.789
top = 1000
true_labels = []
camel_results = []
patient_probabilities = []
print('Loading data')
for i in tqdm(range(0, num), 0, leave=False, ncols=70):
    camel_sm = torch.load(f'{data_path}{pt_path[i]}', map_location=torch.device('cpu'))
    camel_results.append(camel_sm)
    label = int(pt_path[i][0])
    true_labels.append(label)
true_labels = np.array(true_labels)

print('Processing')
predictions = []
for i in tqdm(range(0, num), 0, leave=False, ncols=70):
    camel_sm = camel_results[i]  # Single pt file
    neg_sm = []
    pos_sm = []

    for k in range(0, camel_sm.shape[0]):
        if camel_sm[k].item() < cutoff:
            neg_sm.append(camel_sm[k].item())
        else:
            pos_sm.append(camel_sm[k].item())
    
    if len(pos_sm) > len(neg_sm):
        if pos_sm:
            pred_sm = np.sort(pos_sm, axis=0)[-top:].mean()
        else:
            pred_sm = 0    
        predictions.append(pred_sm)
    else:
        if neg_sm:
            pred_sm = np.sort(neg_sm, axis=0)[:top].mean()
        else:
            pred_sm = 0
        predictions.append(pred_sm)
    
    if np.isnan(predictions).any():
        continue
    label = int(pt_path[i][0])
    patient_id = (pt_path[i].split('_')[1]).split('.pt')[0]
    patient_probabilities.append((patient_id, pred_sm, label))
    
predictions = np.array(predictions)  # Collect all slice prediction values

# Save patient probabilities to Excel (REPLACE WITH YOUR PATH)
patient_prob_df = pd.DataFrame(patient_probabilities, columns=['Patient_ID', 'Probability', 'Label'])
patient_prob_df.to_excel('./patient_probabilities.xlsx', index=False)  # Example path, replace with your actual path

# Generate histogram (REPLACE WITH YOUR PATH)
plt.hist(predictions, bins=20, edgecolor='black')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Histogram of Patient Probabilities')
plt.savefig('./patient_probability_histogram.jpg')  # Example path, replace with your actual path)
plt.close()