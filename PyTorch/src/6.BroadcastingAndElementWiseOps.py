#!/usr/local/bin/python3
import torch

# Element-wise , componenet-wise, point-wise

# If the two tensors have the same shape, we can perform element wise
# operations on them. +-*/ are all element wise operations.
# Returns a tensor filled with random numbers from a uniform
# distribution on the interval [0,1)
t1 = torch.rand((5, 5))
print(t1.shape)

t2 = torch.rand((5, 5))
print(t2.shape)

result = t1 + t2
print(result)

# Broadcasting works however, just like numpy.
result = result - 1
# A filter matrix
filterMatrix = result > 0
print("Filter Matrix is:\n", filterMatrix)
# Will print the values that were bigger than 0.
print("Values bigger than 0:\n", result[filterMatrix])
# For element wise multiplication use *. For casting use result.int(), or double() etc.
print("Values bigger than 0 in their place as a matrix.\n",
      result * filterMatrix.int()
      )

# We can get the same results using the in-built functions
# Greater equals
print("Greater equals to 0:\n", result, "\n", result.ge(0))
# Greater
print("Greater than 0:\n", result.gt(0))

t3 = torch.rand((5, 1))
# Because of Broadcasting, we can do the following operation.
print("Broadcasting works:\n", t3 + t1)
