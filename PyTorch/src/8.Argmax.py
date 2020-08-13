#!usr/local/bin/python3
import torch
import numpy as np

nparray = np.array([
    [1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2],
    [3, 3, 9, 3, 3]
])
#  will create a copy of the numpy array with same type.
t = torch.tensor(nparray)
print(t, t.dtype, nparray.dtype)
# Will cast it and return a copy.
t = t.double()
print(t, t.dtype)

#  Returns the maximum value of the array
print(t.max())

# Returns maximums in along the first axis. returns two tensors.
# one maximum values, second their index along other axis.
print("Max along first axis:\n", t.max(dim=0))

# Returns index of the maximum element. Countring is row first.
# We usually use argmax function on an output layer of a neural network.
print(t.argmax())
# This will result in an error.
# print(t[t.argmax()])
# But following won't.
print(t.flatten()[t.argmax()])
