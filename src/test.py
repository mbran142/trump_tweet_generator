from train import build_and_train_model
from clean import clean_data
from datasets import tweet_dataset, build_tweet_datasets, string_to_tensor, tensor_to_string
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import torch
import pdb
import json

with open('data/parsed_data.json') as file:
    data = json.load(file)

train, val = build_tweet_datasets('data/parsed_data.json')

for i, item in enumerate(train):

    print(f'{i} | {item.shape}')

train_dataloader = DataLoader(train, shuffle = False, batch_size = 64)
for tweet in train_dataloader:

    print(tweet)
    break