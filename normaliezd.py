import torch

pixels = torch.tensor([0.0, 127.5, 255.0])
normalized = pixels / 255.0

print(normalized)