# Learn PyTorch

A hands-on PyTorch learning project based on the official PyTorch Basics tutorial.

## Setup

Create and activate a virtual environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install PyTorch and TorchVision:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Run

Open a notebook in VS Code, select the project's Python environment as its kernel, and run the cells from top to bottom.

## Repository policy

This repository keeps source notebooks and documentation. It intentionally ignores the virtual environment, downloaded dataset, and model checkpoint files because they can be recreated and would make the repository unnecessarily large.