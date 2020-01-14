print("# 1.1.1 Tensors in One Dimension")
import torch
import math
a = torch.tensor([0, 1, 2, 3, 4])
b = a[1]
print(a[1])
print(b.item())
print(a.size())

x = torch.linspace(0, 2*math.pi, 100)
y = torch.sin(x)
print(x)

print("# 1.1.2 2-Dimensional PyTorch Tensors")
a = [[11, 12, 13],
     [21, 22, 23],
     [31, 32, 33]]
A = torch.tensor(a)
print(A.ndimension())
print(A)
# Slice
print(A[0, 0:2])
print(A[1:3, 2])

# Notations
print("test", 1*10**6)
print("test", 516*10**-7 == 0.0000516)
