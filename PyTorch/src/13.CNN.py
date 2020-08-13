#!/usr/local/bin/python3

import torch
# import Neural Network Library
import torch.nn as nn
import torchvision
import torch.transforms as transforms

# We have prepared the data
# Now we will prepare our model (network)

# Some OOP
class Lizard:
    def __init__(self, name):
        self.name = name

    def setName(self, name):
        self.name = name

liz1 = Lizard("Deep")
liz2 = Lizard("Berry")

# Two componets in each NN: 
# 1. Transformation, 2. Collection of Weights
# All the NNs in PyTorch extend the Module class.
# 