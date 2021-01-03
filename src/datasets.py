import torch
from torch.utils.data import Dataset
import numpy as np
import os

# dataset class. Used to obtain total dataset at each epoch of training.
class tweet_dataset(Dataset):

    def __init__(self, file_path):
        tweet_file = open(file_path, 'r')
        self.data = tweet_file.readlines()

        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        return data[item]