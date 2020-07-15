#!/usr/bin/python3

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

print("This is the rank of t: {}".format(len(t.shape)))