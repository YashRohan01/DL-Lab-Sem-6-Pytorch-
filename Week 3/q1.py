import torch
import matplotlib.pyplot as plt

x = torch.tensor(
    [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]
)

y = torch.tensor(
    [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]
)

learningRate = 0.001

b = torch.rand([1],requires_grad=True)
w = torch.rand([1],requires_grad=True)
print(f"w = {w}, b = {b}")

Losses = []

epochs = 8

for i in range(epochs):
    loss = 0
    for j in range(len(x)):
        y_p = w*x[j] + b
        loss += (y[j] - y_p)**2
    loss /= 2*len(x)
    Losses.append(loss.item())
    loss.backward()
    with torch.no_grad():
        w -= learningRate * w.grad
        b -= learningRate * b.grad
    print(f"epoch = {i} ----> Loss = {loss}, w = {w}, b = {b}")
    w.grad.zero_()
    b.grad.zero_()

plt.plot(Losses)
plt.show()







