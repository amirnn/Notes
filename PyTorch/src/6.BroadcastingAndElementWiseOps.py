#!/usr/bin/python3
import torch

# If the two tensors have the same shape, we can perform element wise
# operations on them. +-*/ are all element wise operations.

t1 = torch.rand((5,5))
print(t1.shape)

t2 = torch.rand((5,5))
print(t2.shape)

result = t1 + t2
print(result)

# Broadcasting works however, just like numpy.
result = result - 1
# A filter matrix
filterMatrix = result > 0
print("Filter Matrix is:\n",filterMatrix)
# Will print the values that were bigger than 1.
print("Values bigger than 1:\n",result[filterMatrix])
print("Values bigger t")

# We can get 0 or 1 instead of True or False following the

# Becasue of Broadcasting we can do the following operation.
t3 = torch.rand((5,1))

print(t3 + t1)