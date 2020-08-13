#!/usr/local/bin/python3
import numpy as np
import torch



data = np.array([1,2,3])

t1 = torch.Tensor(data) # Creates a tensor with PyTorch's default type
t2 = torch.tensor(data) # Creates tensor based on the intrinsic type of array

# Print the default dtype that the Tensor constructor uses
print(torch.get_default_dtype())


# This wont work
#  t3 = torch.Tensor(data,dtype=torch.float64)
t3 = torch.tensor(data,dtype=torch.float64)
# Transfer the Tensor to GPU, it does not work on macos.
# t4 = t3.cuda()

# As you can see the .cuda() call does not change the device for the array.
# It initilizes (copies) the same data on gpu only.
print("t3 device: {}".format(t3.device))


# Again.
# These two copy the data.
t1 = torch.Tensor(data)
t2 = torch.tensor(data) #This is our to go function.
# Changes made to data will present itself on t3 and t4
# This is because t3 and t4 are references or mirrors (views).
# This data sharing makes these methods more efficient. (Based on the application.)
t3 = torch.as_tensor(data)  #accepts everything.
t4 = torch.from_numpy(data) #only accepts numpy arrays



