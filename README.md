# Learn PyTorch

My hands-on PyTorch learning project, following the PyTorch Basics tutorial.

## What is included

- Python and tensor practice
- A simple linear-regression project
- FashionMNIST classification with a fully connected neural network
- FashionMNIST classification with a CNN, including train/validation/test separation

## Setup

Create and activate a virtual environment in PowerShell:

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

Install the CUDA-enabled PyTorch build (used with an NVIDIA GPU):

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Run the CNN project

```powershell
python fashion_cnn.py
```

The script downloads FashionMNIST into `data/` when needed, trains a CNN for five epochs, saves the best validation checkpoint locally, and reports final test accuracy.

## Result

Latest run: **91.15% final test accuracy** using an NVIDIA GeForce RTX 4060 Laptop GPU.

## Repository policy

The repository keeps source code and documentation. It intentionally ignores the virtual environment, downloaded dataset, and `.pth` checkpoints because they can be recreated and would make the repository unnecessarily large.
