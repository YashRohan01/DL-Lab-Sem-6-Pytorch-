import torch

b = torch.tensor(2.,requires_grad=True)
x = torch.tensor(3.,requires_grad=True)
w = torch.tensor(0.5,requires_grad=True)

u = w*x
print("u = w*x = ",u)

v = u + b
print("v = u + b = ",v)

a = torch.sigmoid(v)
print("a = sigmoid(v) = ",a)

a.backward()

print("da/db = ",b.grad)
print("da/dx = ",x.grad)
print("da/dw = ",w.grad)