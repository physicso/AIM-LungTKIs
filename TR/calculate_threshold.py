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

def roc(y_true, y_score, sklearn=True):
    pos_label = 1
    # Count the number of positive and negative samples
    num_positive_examples = (y_true == pos_label).sum()  # Number of positive samples
    num_negative_examples = len(y_true) - num_positive_examples

    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score) + 1

    # Calculate FPR and TPR based on sorted prediction scores
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
    cutoff = thresholds[maxindex]  # Best cutoff threshold
    index = thresholds.index(cutoff)

    sensitivity = tpr[index]  # Sensitivity
    specificity = 1 - fpr[index]  # Specificity

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
    accuracy = correct / y_score.shape[0] * 100

    return auc, sensitivity, specificity, index, fpr, tpr, cutoff, accuracy

def auc_figure(auc, sensitivity, specificity, index, fpr, tpr, cutoff, accuracy):
    fig, ax = plt.subplots()
    plt.plot([0, 1], '--')
    plt.plot(fpr[index], tpr[index], 'bo')
    ax.text(fpr[index], tpr[index] + 0.02, f'cut_off={round(cutoff, 8)}', fontdict={'fontsize': 10})
    plt.plot(fpr, tpr)
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    text = f'AUC:{round(auc, 3)}\nSE:{round(sensitivity,3)}\nSP:{round(specificity,3)}\nAccuracy:{round(accuracy, 3)}%\n'
    ax.text(0.6, 0.05, text, fontsize=12)

    # Save figure (REPLACE WITH YOUR PATH)
    plt.savefig('./train_AUC.png')   

# Data path (REPLACE WITH YOUR DATA PATH)
data_path = './train/'    
pt_path = os.listdir(data_path)
num = len(pt_path)

auc_best = 0
cutoff = 0.534
top = 50
threshold = range_threshold(start=0, end=cutoff, step=0.01)
true_labels = []
camel_results = []
patient_probabilities = []  # Modified variable

print('Loading data')
for i in tqdm(range(0, num), 0, leave=False, ncols=70):
    camel_sm = torch.load(f'{data_path}{pt_path[i]}', map_location=torch.device('cpu'))
    camel_results.append(camel_sm)
    label = int(pt_path[i][0])
    true_labels.append(label)
true_labels = np.array(true_labels)

print('Processing')
for j in range(0, len(threshold)):
    print(f'{j}/{len(threshold)}')
    predictions = []
    patient_ids = []
    for i in tqdm(range(0, num), 0, leave=False, ncols=70):
        name = (pt_path[i].split('_')[1]).split('.pt')[0]
        patient_ids.append(name)
        camel_sm = camel_results[i]  # Single pt file
        neg_sm = []
        pos_sm = []

        for k in range(0, camel_sm.shape[0]):
            if camel_sm[k].item() < (cutoff - threshold[j]):
                neg_sm.append(camel_sm[k].item())
            elif camel_sm[k].item() >= (cutoff + threshold[j]):
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

    predictions = np.array(predictions)
    patient_ids = np.array(patient_ids)
    if np.isnan(predictions).any():
        continue
    auc, se, sp, index, fpr, tpr, cutoff, acc = roc(true_labels, predictions)
    print(f'AUC:{round(auc,5)}, Threshold:{threshold[j]}')
    patient_name = (pt_path[i].split('_')[1]).split('.pt')[0]
    patient_probabilities.append((patient_name, predictions))
    if auc > auc_best:
        print(f'>>>>>>Best AUC:{round(auc,5)}, Threshold:{threshold[j]}<<<<<<')
        auc_figure(auc, se, sp, index, fpr, tpr, cutoff, acc)
        auc_best = auc
        np.save('true.npy', true_labels)
        np.save('pred.npy', predictions)
        np.save('id.npy', patient_ids)

true_labels = np.load('true.npy')
predictions = np.load('pred.npy')
patient_ids = np.load('id.npy')
data = pd.DataFrame({
    'True': true_labels,
    'Pred': predictions,
    'ID': patient_ids
})

# Save to Excel (REPLACE WITH YOUR PATH)
data.to_excel('./train.xlsx', index=False)