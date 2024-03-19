import torch.nn
import torch.nn as nn


def to_gfn(module):
    module.log_Z = nn.Parameter(torch.zeros(1))

# class
