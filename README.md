This repository demonstrates the implementation of basic neural networks using:

🔹 TensorFlow (Keras API)

🔹 PyTorch

🔹 NumPy

🔹 Matplotlib

The goal of this project is to understand how neural networks learn simple mathematical relationships and perform basic classification tasks.

📌 Project Overview

This project covers:

✅ Linear Regression using Neural Networks (y = 2x)

✅ Binary Classification

✅ Loss Curve Visualization

✅ Accuracy Tracking

✅ Comparison between TensorFlow and PyTorch workflows

📦 Installation

Install required dependencies:

pip install tensorflow torch matplotlib numpy

🧠 Project 1: Linear Regression (y = 2x)

We train a simple neural network to learn the relationship:

𝑦
=
2
𝑥
y=2x
Dataset
Input (x):  [1, 2, 3, 4, 5]
Output (y): [2, 4, 6, 8, 10]


The model learns:

output = weight × input + bias

🔹 TensorFlow Implementation

Built using tf.keras.Sequential

Optimizer: Stochastic Gradient Descent (SGD)

Loss: Mean Squared Error (MSE)

Trained for 200 epochs

Visualized training loss curve

Key Learning

How weights update during training

How loss decreases over epochs

How prediction works after training

Example prediction:

Input: 10
Output: ≈ 20

🔹 PyTorch Implementation

Model created using nn.Module

Layer: nn.Linear(1,1)

Loss Function: MSELoss

Optimizer: SGD

Manual training loop

Loss plotted across epochs

Key Learning

How .backward() computes gradients

How optimizer.step() updates parameters

Difference between TensorFlow and PyTorch training style

🧠 Project 2: Binary Classification

We classify numbers into two categories:

Class 0 → Small numbers (1,2,3)

Class 1 → Large numbers (6,7,8)

Model Details

Activation Function: Sigmoid

Loss Function: Binary Crossentropy

Metric: Accuracy

Epochs: 100

The accuracy improves as training progresses, showing how the model learns decision boundaries.

📊 Visualizations Included

📉 Training Loss Curve (Regression)

📈 Training Accuracy Curve (Classification)

📉 PyTorch Loss Curve

🎯 Concepts Covered

Neural Networks basics

Weights & Bias

Gradient Descent

Loss Functions

Regression vs Classification

Activation Functions

Training Loops

Model Evaluation

⚠️ Important Notes

Accuracy metric is used only for classification.

For regression problems, use Mean Squared Error.

Modern TensorFlow practice uses tf.keras.Input() instead of passing input_shape directly.

🚀 Future Improvements

Add multi-layer neural network

Implement GPU training

Add real dataset (MNIST / CIFAR-10)

Save & load trained models

Compare performance benchmarks

🛠 Technologies Used

Python 3.x

TensorFlow 2.x

PyTorch

NumPy

Matplotlib

📌 Purpose of This Repository

This project was created to build strong foundational understanding of:

Deep Learning frameworks

Model building workflow

Training mechanics

Practical implementation differences between TensorFlow & PyTorch
