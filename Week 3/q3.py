import torch
import matplotlib.pyplot as plt



class RegressionModel:
    def __init__(self):
        self.w = torch.tensor([1.0], requires_grad=True)
        self.b = torch.tensor([1.0], requires_grad=True)
    def forward(self,x):
        y_p = self.w*x + self.b
        return y_p
    def update(self,alpha):
        self.w -= alpha * self.w.grad
        self.b -= alpha * self.b.grad
    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()
    def criterion(self,y,y_p):
        loss = (y_p - y)**2
        return loss

model = RegressionModel()

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

alpha = 0.001

Losses = []
epochs = 15

for i in range(epochs):
    loss = 0
    for j in range(len(x)):
        y_p = model.forward(x[j])
        loss += model.criterion(y[j],y_p)
    loss /= len(x)
    Losses.append(loss.item())
    loss.backward()
    with torch.no_grad():
        model.update(alpha)
    model.reset_grad()
    print(f"epoch = {i} ------> Loss = {loss}, w = {model.w}, b = {model.b}")

plt.plot(Losses)
plt.show()