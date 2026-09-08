## CIFAR-10 Image Classification: ResNet vs Vision Transformer (ViT)

This project benchmarks two deep learning architectures — ResNet (Convolutional Neural Network) 
and Vision Transformer (ViT) — on the CIFAR-10 image classification dataset.

## Overview
CIFAR-10 consists of 60,000 32×32 color images across 10 classes. This project trains and 
evaluates both models to compare their performance, convergence behavior, and generalization ability.

## Models
- **ResNet** – Deep residual network leveraging skip connections for efficient gradient flow
- **ViT (Vision Transformer)** – Patch-based transformer model applying self-attention to image tokens

## Results
| Model  | Test Accuracy |
|--------|--------------|
| ResNet |    80.65%    |
| ViT    |    65.74%    |

## Tech Stack
- Python, PyTorch
- torchvision, Transformers
- Matplotlib (training curves)

## Key Insights
- ResNet converges faster on small datasets like CIFAR-10
- ViT requires more data/compute but captures global context via attention
- Trade-off between inductive bias (CNN) vs learned representations (Transformer)
