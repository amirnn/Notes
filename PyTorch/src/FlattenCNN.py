#!/usr/bin/python3
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
t = torch.stack((t1,t2,t3))
print("Shape of stacked tensor:\n",t.shape)

# Will print ([3,4,4]) The 3 is the batch size, 
# the 4 and 4 are width and height.

# To add color chanel representation we add one other axis after 
# the batch size.  3,1,4,4
t = t.reshape(3,1,4,4) 

# first image in batch
print("first image in batch:\n",t[0])

# first color channel in first image in batch
print("first color channel in first image in batch:\n",t[0][0])

# first row of pixels in first color channel in the first image in the batch
print("first row of pixels in first color channel in the first image in the batch:\n",t[0][0][0])

# first pixel in above
print("first pixel in above:\n",t[0][0][0][0])