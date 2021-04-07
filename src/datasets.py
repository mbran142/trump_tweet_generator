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

    # convert to tensor format
    tweet_data = [string_to_tensor(tweet) for tweet in tweet_data]

    size = len(tweet_data)

    indices = np.arange(len(tweet_data))
    np.random.shuffle(indices)

    val_i = indices[: int(size * ratio)]
    train_i = indices[int(size * ratio) :]

    val = Subset(tweet_data, val_i)
    train = Subset(tweet_data, train_i)

    return train, val


# convert string into tensor. Values go from 0-31.
def string_to_tensor(input):

    # encoding:
    # [a-z] = 0-25
    # space = 26
    # period = 27       ! and ? are replaced with .
    # comma = 28
    # '#' symbol = 29   all digits are replaced with #
    # <start> = 30
    # <pad> = 31        tweets are padded to length 290

    out_list = [ 30 ]

    for char in input:

        if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            out_list.append(ord(char) - ord('A'))

        elif char in 'abcdefghijklmnopqrstuvwxyz':
            out_list.append(ord(char) - ord('a'))

        elif char == ' ':
            out_list.append(26)

        elif char in '.!?':
            out_list.append(27)

        elif char == ',':
            out_list.append(28)

        elif char in '0123456789':
            out_list.append(29)

        elif char == '&':
            out_list += [0, 13, 3]  # 'and'

    padding = [ 31 ] * (290 - len(out_list))

    out_list += padding
    return torch.tensor(out_list)


# convert tensor to string based on above format
def tensor_to_string(input):

    output = ''

    for val in input:

        if val < 26:
            output += chr(val + ord('a'))

        elif val == 26:
            output += ' '

        elif val == 27:
            output += '.'

        elif val == 28:
            output += ','

        elif val == 29:
            output += '#'

        elif val == 31:
            break

    return output