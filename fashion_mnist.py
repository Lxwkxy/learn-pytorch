import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

# print(len(training_data))

# image, label = training_data[0]
# print(image.shape)
# print(label)

# from torch.utils.data import DataLoader

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

# batch_size = 64

# # train_dataloader = DataLoader(
# #     training_data, batch_size=batch_size, shuffle=True
# # )

# test_dataloader = DataLoader(
#     test_data, batch_size=batch_size
# )

# x_batch, y_batch = next(iter(train_dataloader))

# print(x_batch.shape)
# print(y_batch.shape)

from torch import nn

# model = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(28 * 28, 512),
#     nn.ReLU(),
#     nn.Linear(512, 10)
# )

# logits = model(x_batch)
# print(logits.shape)

# loss_fn = nn.CrossEntropyLoss()
# loss = loss_fn(logits, y_batch)

# print(loss.item())

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# def train_one_epoch(dataloader, model, loss_fn, optimizer):
#     model.train()

#     for x_batch, y_batch in dataloader:
#         logits = model(x_batch)
#         loss = loss_fn(logits, y_batch)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     return loss.item()

# print(train_one_epoch(train_dataloader, model, loss_fn, optimizer))

# def test_accuracy(dataloader, model):
#     model.eval()

#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for x_batch, y_batch in dataloader:
#             logits = model(x_batch)
#             predicted = logits.argmax(dim=1)

#             correct += (predicted == y_batch).sum().item()
#             total += y_batch.size(0)

#     return correct / total

# print(test_accuracy(test_dataloader, model))

# for epoch in range(5):
#     loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
#     print(f"epoch {epoch}: loss = {loss}")

# print(test_accuracy(test_dataloader, model))

# torch.save(model.state_dict(), "fashion_model.pth")

loaded_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

loaded_model.load_state_dict(
    torch.load("fashion_model.pth", weights_only=True)
)

loaded_model.eval()

# print(test_accuracy(test_dataloader, loaded_model))


image, true_label = test_data[0]

print(image.shape)
print(true_label)

image_batch = image.unsqueeze(0)
print(image_batch.shape)

with torch.no_grad():
    logits = loaded_model(image_batch)
    predicted_label = logits.argmax(dim=1).item()

print(predicted_label)
print(true_label)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

predicted_name = class_names[predicted_label]
true_name = class_names[true_label]

print("Predicted:", predicted_name)
print("Actual:", true_name)

probabilities = torch.softmax(logits, dim=1)
confidence = probabilities[0, predicted_label].item()

print(confidence)

top_probabilities, top_labels = torch.topk(
    probabilities, k=3, dim=1
)

for probability, label in zip(top_probabilities[0], top_labels[0]):
    print(class_names[label.item()], probability.item())