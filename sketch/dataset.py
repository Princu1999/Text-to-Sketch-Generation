from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, npy_files,data_type='train', Nmax=100):
        self.data = []
        self.lengths = []
        for file in npy_files:
            raw_data = np.load(file,encoding='latin1', allow_pickle=True)
            for sketch in raw_data[data_type]:
                processed = preprocess_sketch(sketch)
                padded, length = pad_sketch(processed, Nmax)
                self.data.append(padded)
                self.lengths.append(length)
        self.data = np.array(self.data)
        self.lengths = np.array(self.lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.lengths[idx]




def preprocess_sketch(sketch):
    """ Convert (dx, dy, pen_status) to (∆x, ∆y, p1, p2, p3) format. """
    dx, dy, pen_status = sketch[:, 0]/50, sketch[:, 1]/50, sketch[:, 2]
    p1 = (pen_status == 0).astype(np.float32)
    p2 = (pen_status == 1).astype(np.float32)
    p3 = (pen_status == 2).astype(np.float32)
    return np.stack([dx, dy, p1, p2, p3], axis=1)



def pad_sketch(sketch, Nmax):
    """ Pad sketches to length Nmax as per Sketch-RNN paper. """
    padded = np.zeros((Nmax, 5), dtype=np.float32)
    length = min(len(sketch), Nmax)
    padded[:length] = sketch[:length]
    padded[length:] = [0, 0, 0, 0, 1]  # End of sequence marker
    return padded, length