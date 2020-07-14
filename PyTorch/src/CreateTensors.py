#!/usr/bin/python3

import numpy as np
import torch as t

data = np.array([1,2,3])
print(type(data), data.dtype)

#  A constructor
tensorVarNotMatching = t.Tensor(data)
print(tensorVarNotMatching, tensorVarNotMatching.dtype)

#  A factory function, Notice the small t in tensor
tensorVarMatching = t.tensor(data)
print(tensorVarMatching, tensorVarMatching.dtype)

# Also a factory funtion
tensorVarMatching = t.as_tensor(data)
print(tensorVarMatching, tensorVarMatching.dtype)

# also
tensorVarMatching = t.from_numpy(data)
print(tensorVarMatching, tensorVarMatching.dtype)


# Helpful functions.
print(t.eye(5))
print(t.zeros(2,2))
print(t.ones(2,2))
print(t.rand(5,5))
