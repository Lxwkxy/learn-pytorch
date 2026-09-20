import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x

y.backward()

print(x.grad)