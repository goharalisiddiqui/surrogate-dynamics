from scipy.stats import multivariate_normal
from statistics import mean as list_mean
from matplotlib import rc
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning.loggers as pl_loggers
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.distributions.normal import Normal
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


class FlatEmb(pl.LightningModule):
    def __init__(self,
                 outname: str = './untitled/untitled_'):
        super().__init__()
        self.flattern = nn.Flatten(start_dim=1)
        self.unflatten = nn.Unflatten(1, (-1, 3))

    def forward(self, x):
        return x

    def get_latent(self, x):
        x = self.flattern(x)
        return x, None  # None is for the covariance matrix

    def get_latent_mean(self, x):
        x = self.flattern(x)
        return x

    def decode_latent(self, x):
        return self.unflatten(x)
