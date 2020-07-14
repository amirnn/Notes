#!/usr/bin/python3

import torch

print("PyTorch Version: {0}".format(torch.__version__))
if torch.cuda.is_available():
    print("CUDA version {0} is available.".format(torch.version.cuda))



