import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths (replace with your own if needed)
INFERENCE_DIR = './outputs/test_inference/'
LABEL_CSV_PATH = './data/cli_label_PFS.xlsx'
EXPORT_RESULT_PATH = './results/patient_level_results.xlsx'
EXPORT_PLOT_PATH = './results/patient_error_histogram.jpg'

os.makedirs('./results', exist_ok=True)

# Load labels and patient IDs
df = pd.read_excel(LABEL_CSV_PATH)
patient_ids = df['PatientID']

# Aggregate predictions per patient
patient_results = {}
file_list = os.listdir(INFERENCE_DIR)

for fname in file_list:
    label, identifier = fname.split('_')[0], fname.split('_')[1]
    identifier = identifier.split('.')[0].lstrip('0')
    prediction_tensor = torch.load(os.path.join(INFERENCE_DIR, fname), map_location='cpu')
    prediction_mean = prediction_tensor.mean().item()

    if identifier not in patient_results:
        patient_results[identifier] = [(float(label), prediction_mean)]
    else:
        patient_results[identifier].append((float(label), prediction_mean))

# Compute per-patient average error
results, patient_errors = [], []
for pid, data in patient_results.items():
    labels = np.array([x[0] for x in data])
    preds = np.array([x[1] for x in data])
    error = np.mean(np.abs(preds - labels))
    results.append((pid, labels.tolist(), preds.tolist(), error))
    patient_errors.append(error)

avg_error = np.mean(patient_errors)
print(f"Average Patient-Level Error: {avg_error:.4f}")

# Save to Excel
df_out = pd.DataFrame(results, columns=['PatientID', 'Labels', 'Predictions', 'PFS_Error'])
df_out.to_excel(EXPORT_RESULT_PATH, index=False)

# Plot histogram
plt.hist(patient_errors, bins=20, edgecolor='black')
plt.xlabel('PFS Error')
plt.ylabel('Patient Count')
plt.title('Distribution of Patient-Level PFS Prediction Error')
plt.tight_layout()
plt.savefig(EXPORT_PLOT_PATH)
plt.close()
