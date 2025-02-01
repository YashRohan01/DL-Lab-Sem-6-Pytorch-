import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class LinearDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.w * x + self.b

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

dataset = LinearDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = RegressionModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 150
Losses = []

for epoch in range(epochs):
    epoch_loss = 0.0
    for data in dataloader:
        inputs, targets = data

        optimizer.zero_grad()

        y_pred = model(inputs)

        loss = criterion(y_pred, targets)
        epoch_loss += loss.item()

        loss.backward()

        optimizer.step()

    Losses.append(epoch_loss / len(dataloader))

    print(f"Epoch = {epoch+1} ----> Loss = {epoch_loss / len(dataloader)}")

# Plot the loss curve
plt.plot(Losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve During Training')
plt.show()

# Print final model parameters
print(f"Trained w = {model.w.item()}, b = {model.b.item()}")
