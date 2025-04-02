
# PFS Time Prediction from Pathology Images and Clinical Data

This repository provides a pipeline for predicting **Progression-Free Survival (PFS)** time in cancer patients, using **whole-slide pathology image patches** and **clinical data**. The model leverages an MLP (Multilayer Perceptron) regression approach to predict the PFS time (in months).

## Key Features

- **Hybrid Feature Extraction**: Combines image features extracted from pathology slides with clinical data (e.g., patient demographics and medical history).
- **MLP Regression Model**: Predicts PFS time based on fused features from pathology images and clinical data.
- **Patch-Level and Patient-Level Inference**: Performs inference at both the **patch** level (individual image segments) and **patient** level (aggregating multiple patches).
- **Evaluation**: Provides MAE-style error metrics and visualizations, such as histograms of PFS prediction errors.

## Directory Structure

The repository contains the following files:

```
.
├── extract_image_clinical_features.py      
├── train_mlp_pfs.py                        
├── inference_mlp_patch.py                  
├── aggregate_patch_to_patient.py           
├── models.py                               
├── README.md                              
```

## Quick Start Guide

To get started with this project, follow these steps:

### 1. Install Dependencies

This project requires Python 3.8+ and several Python libraries. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### 2. Extract Image and Clinical Features

Run the following script to extract features from pathology slides and clinical data:

```bash
python extract_image_clinical_features.py
```

### 3. Train the MLP Regressor

Use the following script to train the MLP model that will predict PFS time based on the extracted features:

```bash
python train_mlp_pfs.py
```

### 4. Perform Inference on Patch-Level Data

After training, you can use the trained model to run inference on individual patches (image segments). This will predict the PFS time for each patch:

```bash
python inference_mlp_patch.py
```

### 5. Aggregate Patch Predictions to Patient-Level

Finally, you can aggregate the patch-level predictions to obtain the final patient-level PFS prediction:

```bash
python aggregate_patch_to_patient.py
```

## Dependencies

This project relies on the following libraries:
- Python 3.8+
- PyTorch 1.10+
- OpenCV
- pandas
- tqdm
- matplotlib
- torchvision

Install them with:

```bash
pip install -r requirements.txt
```

## Citation

If you use this repository in your research, please cite the following:

- (Citation details coming soon)

## License

This project is licensed under the MIT License.
