import torch
import matplotlib.pyplot as plt

x = torch.tensor(
    [2,4]
)

y = torch.tensor(
    [20,40]
)

w = torch.tensor([1.0],requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

alpha = 0.001
Losses = []

for i in range(250):
    loss = 0
    for j in range(len(x)):
        y_p = w*x[j] + b
        loss += (y_p - y[j])**2
    loss /= 2*len(x)
    Losses.append(loss.item())
    loss.backward()
    with torch.no_grad():
        w -= alpha*w.grad
        b -= alpha*b.grad
    w.grad.zero_()
    b.grad.zero_()
    print(f"epoch = {i} -----> Loss = {loss}, w = {w}, b = {b}")

plt.plot(Losses)
plt.show()