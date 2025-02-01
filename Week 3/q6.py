import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = torch.tensor([[3.0,8.0], [4.0,5.0], [5.0,7.0], [6.0,3.0],[2.0,1.0]])
y = torch.tensor([-3.7,3.5,2.5,11.5,5.7])

model = nn.Linear(2,1)
criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

Losses = []
epochs = 100

for i in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    Losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f"epoch = {i} -----> Loss = {loss}")

plt.plot(Losses)
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.show()

input_data = torch.tensor([[3.0, 2.0]])  # Input values (X1 = 3, X2 = 2)

model.eval()
with torch.no_grad():  # Disable gradient calculation for inference
    predicted_y = model(input_data)  # Get the model's prediction

print(f"Prediction for X1 = 3 and X2 = 2: {predicted_y.item()}")
