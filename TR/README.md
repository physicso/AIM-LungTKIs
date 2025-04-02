
# TR Classification Project

## Project Overview

This is a medical image classification project aimed at classifying pathology images using deep learning models. The project utilizes ResNet50 for feature extraction and passes the features to a Multi-Layer Perceptron (MLP) for classification. Additionally, the project implements a dual-threshold voting strategy for inferring from patch-level to slice-level and generates heatmaps for intuitive visualization of classification results.

## File Structure

```
project/
├── efficient_net.py
├── extract_features.py
├── train_mlp_classifier.py
├── train_mlp_features.py
├── inference_to_slices.py
├── classify_wsi.py
├── model.py
├── mixture_of_experts.py
├── aggregate_to_patient.py
├── run_camel_pipeline.py
├── threshold_criterion.py
├── calculate_threshold.py
└── README.md
```

## File Descriptions

### `calculate_threshold.py`
- **Functionality**: Implements ROC curve calculation and dual-threshold voting strategy for inferring from patch-level predictions to slice-level.
- **Main Features**:
  - Computes ROC curve and AUC values.
  - Implements dual-threshold voting strategy to generate final slice-level prediction results.
  - Plots ROC curve.

### `thresholding.py`
- **Functionality**: Performs inference on classified slices to generate prediction probabilities for each slice.
- **Main Features**:
  - Loads pre-trained models.
  - Infers on each slice's patches to generate prediction probabilities.
  - Saves prediction results as `.pt` files.

### `run_camel_pipeline.py`
- **Functionality**: Trains a classification model using a pre-trained ResNet50.
- **Main Features**:
  - Defines training parameters and data paths.
  - Initializes the model and performs distributed training.
  - Saves the trained model weights.

### `aggregate_to_patient.py`
- **Functionality**: Aggregates slice-level prediction results to the patient level.
- **Main Features**:
  - Reads slice-level prediction results.
  - Aggregates the results of the same patient's slices to generate patient-level prediction results.
  - Saves aggregated results as `.pt` files.

### `mixture_of_experts.py`
- **Functionality**: Implements the Mixture of Experts (MoE) model for feature fusion and classification.
- **Main Features**:
  - Defines the structure of the MoE model.
  - Implements expert networks and gating mechanisms.
  - Provides the forward propagation of the MoE model.

### `model.py`
- **Functionality**: Defines the MLP models used for classification.
- **Main Features**:
  - Defines the structure of MLPRegressor and Dino_Mlp models.
  - Provides the forward propagation of the models.

### `main_classify_WSI_hotmap.py`
- **Functionality**: Classifies WSI images and generates heatmaps.
- **Main Features**:
  - Loads pre-trained models.
  - Infers on WSI images to generate heatmaps.
  - Saves heatmaps and classification results.

### `main_camel_WSI_hotmap.py`
- **Functionality**: Classifies WSI images and generates heatmaps.
- **Main Features**:
  - Defines training and testing procedures.
  - Loads data and performs training or testing.
  - Saves trained model weights and testing results.

### `inference_to_slices.py`
- **Functionality**: Performs inference on slices to generate prediction results for each slice.
- **Main Features**:
  - Loads pre-trained models.
  - Infers on each slice's patches to generate prediction results.
  - Saves prediction results as `.pt` files.

### `train_mlp_classifier.py`
- **Functionality**: Trains an MLP model using extracted features.
- **Main Features**:
  - Loads feature data.
  - Defines and trains an MLP model.
  - Saves the trained model weights.

### `train_mlp_features.py`
- **Functionality**: Trains an MLP model using extracted features.
- **Main Features**:
  - Loads feature data.
  - Defines and trains an MLP model.
  - Saves the trained model weights.

### `extract_features.py`
- **Functionality**: Extracts features from images.
- **Main Features**:
  - Loads pre-trained models.
  - Infers on images to extract features.
  - Saves extracted features as `.pt` files.

### `efficient_net.py`
- **Functionality**: Defines the structure of the EfficientNet model.
- **Main Features**:
  - Defines different versions of EfficientNet (e.g., EfficientNet-V2-T, S, M, L).
  - Provides the forward propagation of the model.

## Usage

### Environment Dependencies
- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- pandas
- matplotlib
- tqdm

### Installing Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib tqdm
```

### Data Preparation
- Place the training and testing data in the specified directories.
- Ensure that the data paths match those in the code.

### Training the Model
```bash
python run_camel_pipeline.py
```

### Feature Extraction
```bash
python extract_features.py
```

### Classification Training
```bash
python train_mlp_classifier.py
```

### Inference
```bash
python inference_to_slices.py
```

### Generating Heatmaps
```bash
python classify_wsi.py
```

## Contributing
- If you have any suggestions or improvements for the project, feel free to submit a Pull Request or Issue.

## Contact
- If you have any questions, please contact me at [your_email@example.com](mailto:your_email@example.com).

---
