import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import copy


# 1. เลือก device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


# 2. โหลดข้อมูล
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_data, validation_data = random_split(
    training_data,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42),
)

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)

validation_dataloader = DataLoader(
    validation_data,
    batch_size=batch_size,
    shuffle=False
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
)

# 3. ฟังก์ชัน train และ test
def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()

    total_loss = 0
    batch_count = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count


def test_accuracy(dataloader, model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            predicted = logits.argmax(dim=1)

            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


# 4. สร้าง CNN
first_block = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
)

second_block = nn.Sequential(
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
)

classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

model = nn.Sequential(
    first_block,
    second_block,
    classifier,
).to(device)

print("Model device:", next(model.parameters()).device)


# 5. กำหนด loss และ optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 6. ฝึกและวัดผล
epochs = 5

best_validation_accuracy = 0

for epoch in range(epochs):
    loss = train_one_epoch(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
    )

    validation_accuracy = test_accuracy(validation_dataloader, model)

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(model.state_dict(), "best_fashion_cnn_model.pth")

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"loss = {loss:.4f} | "
        f"validation accuracy = {validation_accuracy:.2%}"
    )

best_model = copy.deepcopy(model)

best_model.load_state_dict(
    torch.load("best_fashion_cnn_model.pth", weights_only=True)
)

best_model = best_model.to(device)
best_model.eval()

final_test_accuracy = test_accuracy(test_dataloader, best_model)
print(final_test_accuracy)