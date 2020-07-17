#!/usr/bin/python3

# 1. prepare the data
# ETL Extraxt, Transform, Load

# 2. build the model

# 3. Train the model

# 4. Analyze the model's results.

import torch
import torchvision
import torchvision.transforms as transforms

# An abstract class for representing a dataset
import torch.utils.data.dataset

# Wraos a dataset and provides access to the underlying data.
import torch.utils.data.dataloader

#  An example implmentation of a Dataset abstract class.
# Two methods are necessary, __getitem__ and __len__
# class OHLC(Dataset):
#     def __init__(self, csv_file):
#         super().__init__()
#         self.data = pd.read_csv(csv_file)
#     def __getitem__(self, index):
#         r = self.data.iloc[index]
#         label = torch.tensor(r.is_up_day, dtype=torch.long)
#         sample = self.normalize(torch.tensor([
#             r.open, r.high, r.low, t.close
#         ]))
#         return sample, label
#     def __len__(self):
#         return len(self.data)

# torchvision provides us with an implmentation of the Dataset class for
# the mnist and fashionmnist databases.
# torchvision > datasets > mnist.py

trainFASHIONSet = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train = True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

trainMNIST = torchvision.datasets.MNIST(
    root="./data/MNIST",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)