# Image Classification using ANN (Fashion-MNIST)

A simple **Artificial Neural Network (ANN)** built with TensorFlow/Keras that classifies 28×28 grayscale clothing images into 10 categories. Developed as part of the CIE-II activity for the Artificial Neural Networks subject, Department of AI & Data Science, Zeal College of Engineering & Research, Pune.

## Problem

Sorting clothing photos by hand is slow. The goal is to give the computer a picture and have it predict the clothing category (for example, Shirt, Sneaker or Bag) automatically.

## Approach

1. Load the Fashion-MNIST dataset (60,000 training and 10,000 test images).
2. Scale pixel values from 0–255 to 0–1 and flatten each image into 784 values.
3. Train a dense ANN with the architecture below.
4. Evaluate on the unseen test set and plot training curves.
5. Predict individual images and show correct/wrong predictions.

### Model architecture

| Layer | Output shape | Parameters |
|---|---|---|
| Flatten | (None, 784) | 0 |
| Dense (128, ReLU) | (None, 128) | 100,480 |
| Dropout (0.2) | (None, 128) | 0 |
| Dense (64, ReLU) | (None, 64) | 8,256 |
| Dense (10, Softmax) | (None, 10) | 650 |
| **Total trainable** | | **109,386** |

Training: Adam optimiser, sparse categorical cross-entropy loss, 10 epochs, batch size 32, 10% of the training data used for validation.

## Results

| Metric | Value |
|---|---|
| Test accuracy (10,000 images) | 87.63% |
| Test loss | 0.3443 |
| Final validation accuracy | 87.78% |

Training accuracy reached about 89.3% and validation accuracy peaked at 88.4% at epoch 4, a mild sign of overfitting. Dropout is used to limit this.

## Dataset

Fashion-MNIST, released by Zalando Research. See [`data/DATASET.md`](data/DATASET.md) for full details, the class list and how it is loaded.

A small sample of 20 test images (2 per class) is included in [`data/sample_test_images/`](data/sample_test_images/) with true labels in `labels.csv`, so you can try the model on your own uploads.

## How to run

### Option 1: Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com) and choose **File → Upload notebook**.
2. Select `ANN_Image_Classification_FashionMNIST.ipynb`.
3. Click **Runtime → Run all**. The dataset downloads automatically. A GPU is not required.
4. To test on your own image, run the last section (Step 10) and upload a photo of a single clothing item, or one of the samples in `data/sample_test_images/`.

### Option 2: Run locally
Requirements: Python 3.9+, TensorFlow 2.x, NumPy, Matplotlib, Jupyter.

```bash
pip install tensorflow numpy matplotlib jupyter
jupyter notebook ANN_Image_Classification_FashionMNIST.ipynb
```

## Repository structure

```
ann-fashion-mnist-classification/
├── README.md
├── ANN_Image_Classification_FashionMNIST.ipynb
└── data/
    ├── DATASET.md
    └── sample_test_images/
        ├── *.png          (20 sample test images)
        ├── labels.csv     (true class for each image)
        └── README.txt
```

## Author

**Om Bodkhe**, T.Y. B.Tech (AI & DS), Zeal College of Engineering & Research, Pune.
Subject faculty: Prof. Hitesh Chaudhari.

## Acknowledgements

- Fashion-MNIST dataset: Zalando Research (https://github.com/zalandoresearch/fashion-mnist)
- TensorFlow / Keras documentation (https://www.tensorflow.org/api_docs)
