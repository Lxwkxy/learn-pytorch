import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

dataloader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=True)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(100):
    for x_batch, y_batch in dataloader:
        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    print(model(torch.tensor([[5.0]])))