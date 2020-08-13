import torch
# import Neural Network Library
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import torch.transforms as transforms

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
# When developing our networks we implement the forward() function.
# Forward stand for forward pass. We use many of the functions
# form nn.functional module file when developing our layer transformations.
# We determine transformations of our layers. These Transformations overall
# makes our networks forward pass functionality. In PyTorch this translates to
# implementing the forward() method for each class(layer,network) that extends
# Module class.
# So:
# 1. Extend the nn.Module base class.
# 2. Define layers as class attributes (members)
# 3. Implement the Network's forward() method.

# First example.


# class Network0:
#     def __init__(self):
#         self.layer = None

#     def forward(self, t):
#         t = self.layer(t)
#         return t

# Compeleting the first example
# Why are we calling super() before calling its __init__ ?
# Shouldn't this be: super.__init__() ?
# In python we call super like this.
# https://docs.python.org/3.9/library/functions.html#super
# https://realpython.com/python-super/
# In Python 3, a call to super() without arguments is equivalent
# to super(B, self) (within methods on class B); note the explicit
# naming of the class. The Python compiler adds a __class__ closure cell
# to methods that use super() without arguments (see Why is Python 3.x's
# super() magic?) that references the current class being defined.
# Look at: https://stackoverflow.com/questions/19776056/the-difference-between-super-method-versus-superself-class-self-method

class Network1(nn.Module):  # line 1
    def __init__(self):
        # In video: line 3
        # super(Network, self).__init__()
        super().__init__()
        self.layer = None

    def forward(self, t):
        t = self.layer(t)
        return t
# At the moment, our Network class has a single dummy layer as an attribute.
# Let’s replace this now with some real layers that come pre-built for us
# from PyTorch's nn library.


class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # implement the forward pass
        return t

# Alright. At this point, we have a Python class called Network that extends PyTorch’s nn.Module class.
# Inside of our Network class, we have five layers that are defined as attributes. We have two convolutional layers,
# self.conv1 and self.conv2, and three linear layers, self.fc1, self.fc2, self.out. We used the abbreviation fc in fc1 and
# fc2 because linear layers are also called fully connected (dense) layers. PyTorch uses the word linear, hence the nn.Linear class name.

# Note that each of our layers (functions extend the Module Class.)
