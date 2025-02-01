import torch

x = torch.tensor(2.5,requires_grad=True)

a = x*x
print("a = x*x = ",a)

b = 2*x
print("b = 2*x = ",b)

c = torch.sin(x)
print("c = sin(x) = ",c)

d = -(a+b+c)
print("d = -(a+b+c) = ",d)

f = torch.exp(d)
print("f = exp(d) = ",f)

f.backward()

print("df/dx = ",x.grad)