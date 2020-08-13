#!/usr/local/bin/python3
import torch

t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])


t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])

t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])

# accepts a tuple as input argument
# Concatenates sequence of tensors along a new dimension. All tensors need to be of the same size.
t = torch.stack((t1,t2,t3))
print("Shape of stacked tensor:\n",t.shape)
# The result will be: torch.Size([3, 4, 4]) The 3 is the batch size, 
# the 4 and 4 are width and height.

# To add color chanel representation we add one other axis after 
# the batch size.  3,1,4,4
# We can achieve the same result by using unsqueeze method as well.
tR = t.reshape(3,1,4,4) 
tU = t.unsqueeze(dim=1)
print(tR.shape, tU.shape)

t = t.reshape(3,1,4,4)
# first image in batch
print("first image in batch:\n",t[0])

# first color channel in first image in batch
print("first color channel in first image in batch:\n",t[0][0])

# first row of pixels in first color channel in the first image in the batch
print("first row of pixels in first color channel in the first image in the batch:\n",t[0][0][0])

# first pixel in above
print("first pixel in above:\n",t[0][0][0][0])

# To Flatten these all work
# ub = t.reshape(1,-1) #This needs squeezing still. torch.Size([1, 48])
# u = t.reshape(1,-1)[0] #? -> [0] is it correct? Yes. 
# t.reshape(-1)
# t.view(t.numel())
# t.view(-1)
# But we use this from PyTorch, It returns a view
t.flatten()
print(t.shape) #The shape is as before.

# We want to flatten each image in every batch seprately.
# We provide the start_dim argument to start flattening from that axis.
# As you can see flatten does not manipulate the original data on memory.
tFlattened = t.flatten(start_dim=1)
print("original t shape after calling the flatten func",t.shape)
print("shape of returned tensor from flatten(start_dim=1): ", tFlattened.shape)

# Also, we could have done the following.
# we can specify the reshape size as: t.reshape(shape=(3,-1))
# Note that in python a 5,6 is as (5,6) or a tuple.
# so reshape(3,-1) is same as reshape((3,-1))
tFlattened = t.reshape(3,-1)
tFlattened2 = t.reshape((3,-1))
print("tFlattened using reshape: ", tFlattened.shape)
print("tFlattened2 using reshape and tuple: ", tFlattened2.shape)