#!/usr/local/bin/python3

import torch
import torchvision
import torchvision.transforms as transforms

# Some additional imports to better understand the data
import numpy as np
import matplotlib.pyplot as plt
# Nothing important. Just setting printing variables.
torch.set_printoptions(linewidth=120)

# In Compose, notice that we call the ToTensor that is we use ();
# Probably it is a factory object, not a function
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
# Remember trainSet has a __len__ function.
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
# Thankfully our data set is balanced.

# To access the data inside
sample = next(
    iter(trainSet)
)

# print(sample)
# Length will be 2, type will be a tuple
# sample is like ( image, label) pairs
print(
    len(sample),
    type(sample)
)
# image = sample[0], label = sample[1]
# this is called sequence unpacking or list unpacking, or deconstructing object.
image, label = sample

# This works with typing though
# image: torch.Tensor = sample[0]
# label: int = sample[1]

# Type will be torch.Size([1,28,28])
print(image.shape)
# Type will be torch.Size([]) or int
print(type(label), label)

# prints out torch.Tensor
print(type(image))

# # this way
# cmape: color map, squeeze removes the 1 redundant dim.
# plt.imshow(np.squeeze(image),cmap="gray")
# plt.title("Label: {}".format(label))
# plt.show()

# or this way, both works
plt.imshow(image.squeeze(), cmap="gray")
plt.title("Label: {}".format(label))
plt.show()


# To get a batch. Almost Identical to data_set.
batch = next(
    iter(trainLoader)
)
# Length will be 2, type will be a list (array) contiaing\
# two tensors
print(
    len(batch),
    type(batch)
)

# We will have multiple of each
images, labels = batch

# The shape of Image will be torch.Size([10,1,28,28])
print("Images shape size: ", images.shape)
# The shape of labels will be: torch.Size([10])
print("Image Labels size:", labels.shape)

# Make a grid of the images. nrow: number of images in a row.
grid = torchvision.utils.make_grid(
    images,
    nrow=10
)

# We transpose to bring the format to: 
# (height, width , color channel)
# it turns to 3 color channels after make_grid call.
print("Grid size is: ",grid.size())
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(
    grid,
    (1, 2, 0)
))
plt.show()
