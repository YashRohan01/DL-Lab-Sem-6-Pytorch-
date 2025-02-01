import torch
from torch.nn.functional import relu_

b = torch.tensor(2.,requires_grad=True)
x = torch.tensor(3.,requires_grad=True)
w = torch.tensor(0.5,requires_grad=True)

u = w*x
print("u = w*x = ",u)

v = u + b
print("v = u + b = ",v)

a = torch.relu_(v)
print("a = relu(v) = ",v)

a.backward()

print("da/db = ",b.grad)
print("da/dx = ",x.grad)
print("da/dw = ",w.grad)