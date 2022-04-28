import numpy as np
import matplotlib.pyplot as plt
import torch

# Can create a tensor from array

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Create tensor from np array

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Making a new tensor - retains shape and data type

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

# Making a new tensor from a shape

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)

# Attributes of a tensor

randshape = rand_tensor.shape
randtype = rand_tensor.dtype
randdevice = rand_tensor.device # Where it is stored

# Check cuda availability -- and move to cuda/GPU

print(torch.cuda.is_available())
#if torch.cuda.is_available():
#    rand_tensor = rand_tensor.to('cpu')
#    print(rand_tensor.device)

# Tensor indexing and slicing

tensor = torch.ones(4, 4)
print('First Row: ', tensor[0])
print('Last Column ', tensor[:, 0])
tensor[:, 1] = 0
print(tensor)

# Joining tensors

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Tensor arithmetic
# Matrix multiplication (y1 == y2 == y3)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# Element-wise product (z1 == z2 == z3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations (store result in operand)

print(tensor, "\n")
tensor.add_(5)
print(tensor)

