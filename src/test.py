from train import build_and_train_model
from clean import clean_data
from train import tweet_to_tensor
from datasets import tweet_dataset, build_tweet_datasets
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import pdb

# pdb.set_trace()

train, val = build_tweet_datasets('data/parsed_data.json')

train_dataloader = DataLoader(train, shuffle=True, batch_size=1)
for tweet in train_dataloader:
    # print(f'Train tweet ex: {tweet}')
    break

val_dataloader = DataLoader(val, shuffle=True, batch_size=4)
for tweet in val_dataloader:
    # print(f'Val tweet ex: {tweet}')
    print(tweet.shape)
    break
