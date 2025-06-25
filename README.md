
 **Course**: Υπολογιστική Όραση  

This repository contains solutions to two assignments from the University of Ioannina course **Computer Vision**, focusing on image classification using traditional and neural network-based approaches.

---

## 🔹 Assignment 1 – Traditional Classifiers on CIFAR-10

The first assignment explores classic machine learning techniques applied to the **CIFAR-10** image dataset. Each classifier was implemented from scratch using **NumPy**, and evaluated on classification accuracy.

###  Implemented Classifiers:

- **Nearest Mean Classifier**: Uses the average image per class to classify based on minimum Euclidean distance.
- **Nearest Neighbor Classifier**: Compares test images to all training images using L2 norm.
- **Linear Classifier (Least Squares & Regularized)**: Solves a linear system to find weight matrices that separate classes.
- **Linear Classifier with SGD**: Uses Stochastic Gradient Descent to learn weights iteratively.

 File: `assignment1.ipynb`

---

## 🔹 Assignment 2 – Neural Network from Scratch

The second assignment implements a **two-layer fully connected neural network** using only **NumPy**, without deep learning libraries. The network is modular and trained using stochastic gradient descent.

###  Files (Assignment 2):
- `layers.py`: Implements fully connected layers and ReLU activation, with forward/backward methods
- `two_layer_net.py`: Defines the neural network structure and training-related logic
- `train.py`: Loads the CIFAR-10 dataset, handles training loop, and generates accuracy plots


## Run Instructions

```bash
pip install numpy matplotlib jupyter

# For Assignment 1
jupyter notebook assignment1.ipynb

# For Assignment 2
python train.py 
```
## Author
Developed individually by Maria Tsatsari
GitHub: maytsatsari
---
