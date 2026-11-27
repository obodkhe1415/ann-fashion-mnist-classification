# Dataset: Fashion-MNIST

## Overview

| Property | Details |
|---|---|
| Name | Fashion-MNIST |
| Creator | Zalando Research |
| Type | Grayscale images of clothing and accessories |
| Image size | 28 × 28 pixels, 1 channel (grayscale) |
| Pixel values | Integers from 0 (black) to 255 (white) |
| Total images | 70,000 |
| Training set | 60,000 images |
| Test set | 10,000 images |
| Number of classes | 10 (balanced, about 7,000 images per class) |
| Source | https://github.com/zalandoresearch/fashion-mnist |
| Licence | MIT (see the source repository) |

The dataset was designed as a drop-in replacement for the classic handwritten-digit dataset (MNIST), so it uses the same image size and file format but shows clothing instead of digits.

## Classes

| Label | Class name |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## How it is loaded in this project

The dataset is downloaded automatically by Keras; no manual download is needed.

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# x_train: (60000, 28, 28), y_train: (60000,)
# x_test:  (10000, 28, 28), y_test:  (10000,)
```

Preprocessing applied in the notebook: pixel values are divided by 255 to scale them to 0–1. A 10% validation split is taken from the training set during training.

## Citation

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.* arXiv:1708.07747.

## Notes

- The dataset is public and contains no personal or confidential information.
- Images are for benchmarking and education, not for production product classification.
