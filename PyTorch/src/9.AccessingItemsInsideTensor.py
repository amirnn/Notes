# !/usr/bin/python3

import torch
import numpy as np

t = torch.rand((5,5))

# return the mean
tMean = t.mean()
print(tMean)

# Now we can do the same along an axis.
tMean = t.mean(dim=0)
print(tMean)

# Now a numpy array
tNumpy = tMean.numpy()
print(tNumpy, tNumpy.dtype, tMean.dtype)

# or a Python list
tList = tMean.tolist()
print(tList)