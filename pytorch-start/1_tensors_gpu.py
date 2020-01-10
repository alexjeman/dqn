import torch
import numpy as np

print('Construct a 5x3 matrix, uninitialized:')
x = torch.empty(5, 3)
print(x)

print('Construct a randomly initialized matrix:')
x = torch.rand(5, 3)
print(x)

print('Construct a matrix filled zeros and of dtype long:')
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

print('Construct a tensor directly from data:')
x = torch.tensor([5.5, 3])
print(x)

print('Create a tensor based on existing tensor:')
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.rand_like(x, dtype=torch.float)
print(x)

print('Get its size:')
print(x.size())

print('Addition synatx_1:')
y = torch.rand(5, 3)
print(x + y)

print('Addition syntax_2:')
print(torch.add(x, y))

print('Addition providing an output tensor as argument:')
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

print('Addition in-place x to y:')
y.add_(x)
print(y)

print('NumPy-like indexing')
print(x[:, 1])

print('Resizing, if you want to resize/reshape tensor, you can use torch.view')
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

print('f you have a one element tensor, use .item() to get the value as a Python number:')
x = torch.randn(1)
print(x)
print(x.item())

print('Converting a Torch Tensor to a NumPy Array:')
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

print('Converting NumPy Array to Torch Tensor:')
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

print('CUDA Tensors can be moved onto any device using the .to method:')
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
