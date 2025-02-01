import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])
print(x.shape)
print(y.shape)

x = x.view(-1,1)
y = y.view(-1,1)

model = nn.Linear(1,1)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr = 0.001)

epochs = 100
Losses = []

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

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, model(x).detach(), color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
