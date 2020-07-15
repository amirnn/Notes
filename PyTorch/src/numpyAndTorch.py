import numpy as np
import torch



data = np.array([1,2,3])

t1 = torch.Tensor(data)
t2 = torch.tensor(data)

# Prin the default dtype that the Tensor constructor uses
print(torch.get_default_dtype())


# This wont work
#  t3 = torch.Tensor(data,dtype=torch.float64)
t3 = torch.tensor(data,dtype=torch.float64)
t4 = t3.cuda()

# As you can see the .cuda() call does not change the device
# It initilizes the same data on gpu only.
print("t3 device: {}".format(t3.device))


# Again.
# These two copy the data.
t1 = torch.Tensor(data)
t2 = torch.tensor(data) #This is our to go function.
# Changes made to data will present itself on t3 and t4
# This is because t3 and t4 are references or mirrors.
# This data sharing makes these methods more efficient.
t3 = torch.as_tensor(data)  #accepts everything.
t4 = torch.from_numpy(data) #only accepts numpy arrays



