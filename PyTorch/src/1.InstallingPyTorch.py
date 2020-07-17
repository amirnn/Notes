#!/usr/bin/python3

import torch

print("PyTorch Version: {0}".format(torch.__version__))
if torch.cuda.is_available():
    print("CUDA version {0} is available.".format(torch.version.cuda))


# Specify the a device, here for example the first gpu.
device = torch.device('cuda:0')
print(device.type)

t1=torch.tensor([1,2,3])
t2=torch.tensor([1.,2.,3.])
t3=torch.tensor([1,5,7],dtype=torch.float32)

tCuda = torch.tensor([1,5,7],dtype=torch.float32).cuda()
# torch.int64
print(t1.dtype) 
# torch.float32
print(t2.dtype)
print(t3.dtype)
print(tCuda.dtype)
# 
print(t1 + t2)
print(tCuda)
# Some noraml calculation
print(tCuda * 5)

# The following line will fail.
# print(tCuda + t3)