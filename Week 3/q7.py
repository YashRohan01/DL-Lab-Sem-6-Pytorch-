import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
Losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(x)

    loss = criterion(y_pred, y)
    Losses.append(loss.item())

    loss.backward()

    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

plt.plot(Losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

plt.scatter(x.numpy(), y.numpy(), color='blue', label='Data Points')

x_test = torch.linspace(0, 110, 100).view(-1, 1)  # Values from 0 to 110 for plotting
y_test = model(x_test)  # Model's output for the test input
plt.plot(x_test.numpy(), y_test.detach().numpy(), color='red', label='Logistic Regression Curve')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Logistic Regression using PyTorch')
plt.show()

print(f"Learned weight: {model.linear.weight.item()}")
print(f"Learned bias: {model.linear.bias.item()}")
