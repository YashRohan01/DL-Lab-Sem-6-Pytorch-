import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN,self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)

    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = FeedForwardNN()
print(f"Initial paramneters : ")
for name,param in model.named_parameters():
    print(f"{name} - {param.data}")

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.1)

x = torch.tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
y = torch.tensor([[0.0],[1.0],[1.0],[0.0]])

epochs = 20000
Losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    prediction = model(x)
    loss = criterion(prediction,y)
    Losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f"epoch -----> {epoch}, Loss = {loss.item()}")

print("Final Parameters: ")
for name,param in model.named_parameters():
    print(f"{name} - {param.data}")

plt.plot(Losses)
plt.show()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params}")