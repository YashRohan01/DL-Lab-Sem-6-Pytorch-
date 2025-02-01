import torch

x = torch.tensor(3.5,requires_grad=True)

a = 8*x*x*x*x
print("a = 8(x^4) = ",a)

b = 3*x*x*x
print("b = 3(x^3) = ",b)

c = 7*x*x
print("c = 7(x^2) = ",c)

d = 6*x
print("d = 6x = ",d)

f = d + 3
print("f = ",f)

f.backward()

print("df/dx = ",x.grad)