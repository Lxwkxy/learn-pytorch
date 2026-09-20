import torch
from torch.utils.data import TensorDataset, DataLoader

x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=2, shuffle=True)

from torch import nn

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

    if ((epoch + 1) % 25 == 0):
        print(f"epoch {epoch + 1} : loss = {loss.item()}")

model.eval()

with torch.no_grad():
    x_test = torch.tensor([[5.0]], dtype=torch.float32)
    test_prediction = model(x_test)
    print(test_prediction.item())

torch.save(model.state_dict(), "linear_model.pth")

loaded_model = nn.Linear(1, 1)

loaded_model.load_state_dict(
    torch.load("linear_model.pth", weights_only=True)
)

loaded_model.eval()

with torch.no_grad():
    loaded_prediction = loaded_model(x_test)

print(loaded_prediction.item())

print(test_prediction.item())
print(loaded_prediction.item())
