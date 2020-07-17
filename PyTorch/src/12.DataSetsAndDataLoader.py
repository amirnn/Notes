# !/usr/bin/python3

import torch
import torchvision
import torchvision.transforms as transforms

# Some additional imports to better understand the data
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)
# in Compose notice that we call the ToTensor(), probably it is 
# a factory object, not a function
trainSet = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=10,
    shuffle=True
)

print(len(trainSet))

# "train_labels" have been renamed to "targets".
print(trainSet.targets)

# This shows the frequecny of the label's frequency. 
# The training set has to  have the equal number of labels, 
# if it does not, we have to replicate the less common ones 
# to make the training set uniform.

# The Paper: A systematic study of the class imbalance problem 
# in convolutional neural networks. 
print(trainSet.targets.bincount())

# To access the data inside 
sample = next(
    iter(trainSet)
    )

# print(sample)
print(
    len(sample),
    type(sample)
)
# image = sample[0], label = sample[1]
# this is called sequence unpacking or list unpacking, or decosntructing object.
image, label = sample

# This works with typing though
# image:torch.Tensor = sample[0]
# label: int = sample[1]

# prints out torch.Tensor
print(type(image))

# # this way
# plt.imshow(np.squeeze(image),cmap="gray")
# plt.title("Label: {}".format(label))
# plt.show()

# or this way, both works
plt.imshow(image.squeeze(),cmap="gray")
plt.title("Label: {}".format(label))
plt.show()


# To get a batch
batch = next(
    iter(trainLoader)
)
print(
    len(batch),
    type(batch)
)

# We will have multiple of each
images, labels = batch

print("Images shape size: ",images.shape)
print("Image Labels size:",labels.shape)

grid = torchvision.utils.make_grid(
    images,
    nrow=10
)
print(grid.size())
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(
    grid,
    (1,2,0)
))
plt.show()