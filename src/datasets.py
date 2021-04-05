import torch
import json
from torch.utils.data import Dataset, Subset
import numpy as np
import os

# dataset class. Used to obtain total dataset at each epoch of training.
class tweet_dataset(Dataset):

    def __init__(self, tweet_list):
        self.data = tweet_list

        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        return self.data[item]


# build training and validation datasets
def build_tweet_datasets(tweet_filename, ratio = 0.2):

    # load file
    with open(tweet_filename) as file:
        tweet_data = json.load(file)

    for i, tweet in enumerate(tweet_data):
        if len(tweet) > 280:
            print(len(tweet), i, tweet)

    # convert to tensor format
    tweet_data = [[ord(char) for char in tweet] for tweet in tweet_data]
    tweet_data = [tweet + [ 128 ] * (280 - len(tweet)) for tweet in tweet_data]     # append padding to all tweets
    tweet_data = torch.tensor(tweet_data)
    # tweet_data = torch.FloatTensor(tweet_data)

    size = len(tweet_data)

    indices = np.arange(len(tweet_data))
    np.random.shuffle(indices)

    val_i = indices[: int(size * ratio)]
    train_i = indices[int(size * ratio) :]

    val = Subset(tweet_data, val_i)
    train = Subset(tweet_data, train_i)

    return train, val