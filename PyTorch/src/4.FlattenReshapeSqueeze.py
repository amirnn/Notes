#!/usr/local/bin/python3

import numpy as np
import torch


# Reshaping operations
# Element-wise operations
# Reduction operations
# Access operations

t = torch.tensor(
    [
       [1,1,1,1],
       [2,2,2,2],
       [3,3,3,3] 
    ],
    dtype=torch.float32
)

# Two ways to access the size.
print(t.size())
print(t.shape)

# Rank of a Tensor. The Rank represesnts the dimensionality of the Tensor.
print("This is the rank of t: {}".format(len(t.shape)))

# Scalar components of a Tensor i.e. Number of elements
# we can construct a tensor using a torch.Size object (Which is a subclass of the tuple).
# Tensor.prod() calculates the product(multiplication) of the elements.
print(torch.tensor(t.shape).prod())
# s = t.shape
# u = torch.tensor(t.shape)
# print(u[0])
# Same as before
t.numel()

print(torch.tensor(t.shape).prod()," is same as: " , t.numel())

# reshape, not that the axis sizes are factors of 12
# reshape returns a view, so it is free of side effects.
t.reshape(1,12)
t.reshape(2,6)
t.reshape(3,4)
print(t.reshape(1,12))
print(t.reshape(2,6))
print(t.reshape(3,4))

# To get rid of the un-needed extra dimensionality i.e. get rid of axis that are 1
# we use squeeze() method.
# For example here we transform the tensor to a 1 dimensional one.
print("Squeezed tensor: " , t.reshape(1,12).squeeze())

# This shows that all these operations are only views not copies.
# since, the original data is intact.
print("The original tensor after squeeze and reshape: {}".format(t))

# To add back the axis we got rid of previously, we use un-squeeze(dim=n) call
# This adds an axis with length 1 at the dim=n, but note that n can only be rank + 1
print("After un-squeeze dim=0: \n",t.reshape(1,12).squeeze().unsqueeze(dim=0))
print("After un-squeeze dim=1: \n",t.reshape(1,12).squeeze().unsqueeze(dim=1))
# Flattening a Tensor
# We flatten the image to 1D array or a rank 1 tensor.
def flatten(t):
    # Providing -1 as input automatically adjusts the correct dimension for the second dimension.
    return t.reshape(1,-1).squeeze()

print("Our own flattened function: {}".format(flatten(t)))


# To concatenate tensors we use torch.cat