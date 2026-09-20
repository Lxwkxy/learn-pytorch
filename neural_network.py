import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 3),  # 2 features → 3 ค่า
    nn.ReLU(),        # ตัดค่าติดลบให้เป็น 0
    nn.Linear(3, 2),  # 3 ค่า → คะแนนของ 2 คลาส
)

x_batch = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

logits = model(x_batch)

print(x_batch.shape)
print(logits.shape)
print(logits)