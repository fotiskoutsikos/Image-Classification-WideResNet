# Image-Classification-WideResNet

## Project Overview

This Jupyter notebook implements Wide Residual Networks (WRN) for image classification on the CIFAR-10 dataset as part of a university Deep Learning course. The project explores different network architectures, regularization techniques, and data augmentation methods.

## Objectives

1. Implement and evaluate three best-performing WRN architectures from the research paper
2. Examine the effect of dropout regularization
3. Apply mixup data augmentation on CIFAR-C dataset
4. Analyze confidence scores with and without mixup

## Key Components

### 1. Model Architecture Evaluation
- Tested three WRN configurations (WRN-40-10, WRN-28-10, WRN-16-8) as recommended in the paper
- Achieved validation accuracies:
  - WRN-40-10: 65.3%
  - WRN-28-10: 66.0% (best performance)
  - WRN-16-8: 63.8%

### 2. Regularization with Dropout
- Added dropout layers with rates:
  - WRN-28-10: 0.3 dropout rate
  - WRN-16-8: 0.4 dropout rate
- Observed that dropout slightly reduced accuracy in these experiments, suggesting more epochs may be needed

### 3. Data Augmentation with Mixup
- Implemented custom Data Generator (CIFAR_Dataset)
- Compared performance with and without mixup augmentation
- Note: The implementation differs from the official mixup, resulting in smaller (<2%) differences

### 4. Confidence Score Analysis (Bonus)
- Examined softmax predictions with/without mixup
- Visualized differences in model confidence

## Technical Implementation

### Environment Setup
- Python 3.8
- PyTorch 1.10.1
- Torchvision 0.11.2
- Google Colab environment

### Key Functions
- `construct_datasets()`: Creates train/validation loaders
- `WideResNet`: Implements the WRN architecture
- `train_model()`: Handles the training loop
- `test_cifar()`: Evaluates model performance

## Usage Notes

1. Mount Google Drive to access required modules
2. Set proper paths for dataset and model saving
3. Run cells sequentially to avoid errors
4. Graded cells are marked with `#### GRADED CELL ####`

## Results and Observations

1. The WRN-28-10 architecture performed best among tested configurations
2. Deeper networks (more layers) required significantly more computational resources
3. Dropout regularization showed potential benefits that might require more training epochs to become apparent
4. The mixup implementation provided modest improvements in model generalization
