import torch

scores = torch.tensor([
    [80, 92, 75],
    [60, 70, 85]
])

print(scores.shape)
print(scores[1, 2])