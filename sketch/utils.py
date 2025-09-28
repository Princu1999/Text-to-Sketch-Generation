from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def find_key_by_value(input_dict, value):

  keys = []
  for key, val in input_dict.items():
      if val == value:
          keys.append(key)
  return keys


