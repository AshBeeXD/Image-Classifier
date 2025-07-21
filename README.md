# CIFAR-10 Image Classifier with Grad-CAM Visualizations

This project implements an image classification model trained on the CIFAR-10 dataset using PyTorch and ResNet18. The model is served through a Gradio web interface and visualized with Grad-CAM heatmaps for model explainability.

A live demo of the project is available on Hugging Face Spaces:
https://huggingface.co/spaces/ashbeexd/image_prediction

------------------------------------------------------------

## Project Structure
```
image-classifier/
|
├── app/                   # Hugging Face Space interface
|   └── app.py             # Gradio-powered inference interface
|
├── data/                  # CIFAR-10 dataset (loaded via torchvision)
|
├── outputs/               # Saved model + Grad-CAM overlays
|   ├── model.pth
|   └── interpretations/
|
├── src/                   # Source scripts
|   ├── dataset.py         # DataLoader and transforms
|   ├── model.py           # ResNet18 model loader
|   ├── train.py           # Training loop
|   └── evaluate.py        # Evaluation utilities
|
├── notebooks/             # Jupyter notebooks
|   ├── 01_exploration.ipynb
|   ├── 02_train.ipynb
|   └── 03_evaluation.ipynb
|
├── requirements.txt       # Dependencies
└── README.md


```

------------------------------------------------------------

## Model Overview

- Architecture: ResNet18 (pre-trained)
- Dataset: CIFAR-10
  (Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Training Epochs: 10
- Optimizer: Adam
- Learning Rate: 1e-3
- Device: CPU/GPU (PyTorch auto-detection)

Grad-CAM is integrated for visual explanations of model predictions.

------------------------------------------------------------

## Training

Training was executed over 10 epochs with progressive improvements in validation accuracy:

Epoch 1 : Accuracy = 57.28%
Epoch 2 : Accuracy = 68.96%
Epoch 3 : Accuracy = 70.68%
Epoch 4 : Accuracy = 72.18%
Epoch 5 : Accuracy = 75.00%
Epoch 6 : Accuracy = 75.94%
Epoch 7 : Accuracy = 79.70%
Epoch 8 : Accuracy = 81.20%
Epoch 9 : Accuracy = 83.10%
Epoch 10: Accuracy = 83.80%

The best model (83.80% accuracy) was saved at the end of training.

------------------------------------------------------------

## Evaluation

Test Accuracy: 84.02%  
Macro Average F1-Score: 0.84  

Per-class performance:

Class        | Precision | Recall | F1-Score
-------------|-----------|--------|---------
airplane     | 0.81      | 0.89   | 0.85
automobile   | 0.90      | 0.93   | 0.92
bird         | 0.85      | 0.75   | 0.80
cat          | 0.77      | 0.62   | 0.68
deer         | 0.83      | 0.83   | 0.83
dog          | 0.78      | 0.73   | 0.75
frog         | 0.84      | 0.93   | 0.88
horse        | 0.83      | 0.91   | 0.87
ship         | 0.93      | 0.90   | 0.91
truck        | 0.85      | 0.92   | 0.88

- Confusion matrix shows strong diagonal dominance, indicating accurate classification across most classes.

------------------------------------------------------------

## Grad-CAM Visualizations

- Grad-CAM heatmaps were generated for five random test images.
- Heatmaps consistently highlighted object regions (e.g., cat’s face, ship’s hull).
- Visual overlays confirmed predictions were based on relevant regions.
- Grad-CAM overlays are saved as downloadable PNG files within the app and in /outputs/interpretations/.

In the deployed Gradio app:
- Top-3 predictions and probabilities are displayed.
- The most confident (Top-1) prediction is displayed in a Label component.

------------------------------------------------------------

## 🚀 Live Demo

You can interact with the trained model here:

https://huggingface.co/spaces/ashbeexd/image_prediction

Features:
- Upload a CIFAR-10 style image.
- View Top-3 predictions with probabilities.
- Visualize Grad-CAM heatmaps.

------------------------------------------------------------

## 📚 Credits

- PyTorch:
  https://pytorch.org/

- torchvision (CIFAR-10 dataset):
  https://pytorch.org/vision/stable/

- Gradio (Web Interface):
  https://gradio.app/

- Hugging Face Spaces (Deployment):
  https://huggingface.co/spaces

- Captum (Grad-CAM Visualizations):
  https://captum.ai/

- Matplotlib & Seaborn (Visualization Libraries):
  https://matplotlib.org/
  https://seaborn.pydata.org/

All tools and datasets are used under their respective open-source licenses.

------------------------------------------------------------

## Future Work

- Expand support beyond CIFAR-10 classes.
- Improve accuracy using deeper models (e.g., ResNet50).
- Extend Grad-CAM visualization to multiple layers.
- Add batch prediction capability.


