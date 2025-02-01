import torch

a = torch.tensor(3.5, requires_grad=True)
b = torch.tensor(2. ,requires_grad=True)

x = 2*a + 3*b
print("x = 2*a + 3*b = ",x)

y = 5*a*a + 3*b*b*b
print("y = 5*a*a + 3*b*b*b = ",y)

z = 2*x + 3*y
print("z = 2*x + 3*y = ",z)

z.backward()

print("a grad = ",a.grad)
print("b grad = ",b.grad)