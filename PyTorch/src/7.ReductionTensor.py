#!/usr/bin/python3

import torch
import numpy as np

#   Reduction operations.

t = torch.rand((3,5))
# We want to reduce and sum along the frist axis
# We put 0 at that dim=0 or first axis.
print(t.sum(dim=0),t.sum(dim=0).shape)

# Now along the second axis.
# We put 0 at 2nd axis. dim=1
print(t.sum(dim=1),t.sum(dim=1).shape)

# As a personal preference I like to do reshaping so I would not get
# zero order vectors. This will give me always a vector with one column.
# This is a leftover good habit from numpy. Will result in -> torch.Size([5, 1])
# Where before it resulted in -> torch.Size([5])
print(t.sum(dim=0).reshape(-1,1).shape)