# 📄 Comparative Study on Vision Transformer and CNN for Solar Image Classification

## 🔬 Overview

This repository contains the code and models used in our manuscript:

> **"Comparative Study on Vision Transformer and Convolutional Neural Networks for Solar Image Classification"**  
> _Currently under review at Solar Physics._

We systematically compare Vision Transformers (ViT) and Convolutional Neural Networks (CNN) for classifying solar images from the photosphere and chromosphere, using high-resolution datasets from the NVST telescope. Our results demonstrate that ViT outperforms CNN in multi-feature tasks and provides more interpretable attention maps, especially in identifying solar prominences and active regions.

## 📁 Repository Structure

```
├── data/                   # Dataset directory (not included due to privacy)
├── output/                 # Saved model checkpoints
├── makedata.py             # Divides the data proportionally
├── train.py                # Unified training script
├── estimate_model.py       # Evaluation and metrics
├── predic.py               # Predicts images in a test folder and visualizes classification results with probability histograms
├── confusion.py            # Computes confusion matrix and classification metrics (Precision, Recall, F1) on the test set
├── utils.py                # Grad-CAM visualization methods for model interpretability
├── singgrad.py             # Performs Grad-CAM visualization on a single image using ViT and ResNet models
├── requirements.txt        # Environment dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/hxdzysr/solarimageclassification.git
cd solarimageclassification
```

### 2. Set up environment

Recommended using `venv` or `conda`. Then install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Prepare data

Place your image dataset in the `data/` directory, or update the path in `train.py`. The expected format is:

```
data/
├── 01-photo-sunspot/
│   ├── img1.jpg
│   └── ...
├── 02-chrom-prominence/
│   ├── img1.jpg
│   └── ...
```

```bash
# Automatically divide the dataset proportionally into the training set, validation set and test set
python makedata.py
```

### 4. Train a model

```bash
# Train ViT
python train.py --model vit

# Train ResNet
python train.py --model resnet

# Train with 4x data augmentation
python train.py --augment4x aug4x
```

### 5. Evaluate

```bash
# Predict the classification results of the test set
python predict.py 
```
```bash
# Calculate the confusion matrix of the model in the test set
python confusion.py
```
## 📊 Results (from manuscript)
🔷 Vision Transformer (ViT)
| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 1                | 0.96      | 0.96   | 0.96     | 114     |
| 2                | 0.97      | 0.85   | 0.90     | 39      |
| 3                | 0.92      | 0.92   | 0.92     | 49      |
| 4                | 0.93      | 0.99   | 0.96     | 86      |
| 5                | 1.00      | 1.00   | 1.00     | 10      |
| 6                | 0.86      | 0.74   | 0.79     | 42      |
| 7                | 0.67      | 0.93   | 0.78     | 15      |
| 8                | 1.00      | 0.29   | 0.44     | 7       |
| 9                | 0.95      | 0.95   | 0.95     | 168     |
| 10               | 0.77      | 0.78   | 0.78     | 82      |
| 11               | 0.86      | 0.92   | 0.89     | 39      |
| **Accuracy**     |           |        | 0.90     | 651     |
| **Macro Avg**    | 0.90      | 0.85   | 0.85     | 651     |
| **Weighted Avg** | 0.91      | 0.90   | 0.90     | 651     |

🔶 ResNet50
| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 1                | 0.96      | 0.95   | 0.96     | 114     |
| 2                | 0.97      | 0.95   | 0.96     | 39      |
| 3                | 0.88      | 0.88   | 0.88     | 49      |
| 4                | 0.94      | 0.94   | 0.94     | 86      |
| 5                | 1.00      | 1.00   | 1.00     | 10      |
| 6                | 0.75      | 0.79   | 0.77     | 42      |
| 7                | 0.82      | 0.93   | 0.88     | 15      |
| 8                | 0.75      | 0.43   | 0.55     | 7       |
| 9                | 0.89      | 0.96   | 0.93     | 168     |
| 10               | 0.76      | 0.71   | 0.73     | 82      |
| 11               | 0.94      | 0.85   | 0.89     | 39      |
| **Accuracy**     |           |        | 0.89     | 651     |
| **Macro Avg**    | 0.88      | 0.85   | 0.86     | 651     |
| **Weighted Avg** | 0.89      | 0.89   | 0.89     | 651     |

## 📌 Notes

- The dataset is not included in this repo due to copyright constraints. Contact the authors for access or use your own labeled solar dataset.
- The code is designed to be flexible for other solar image tasks as well.

## 📚 Citation

If you find this work useful, please cite the paper once published:

```bibtex
@article{Ao25,
  title={Comparative Study on Vision Transformer and Convolutional Neural Networks for Solar Image Classification},
  author={Y.A and D.C},
  journal={Solar Physics},
  year={2025},
  note={Under Review}
}
```

## 🧑‍💻 Contact

For questions or collaborations:

- 📧 Email: aoyuchen@ynao.ac.cn
