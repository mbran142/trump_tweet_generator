from datasets import tweet_dataset
from model import tweet_model
import json
import pdb

def build_and_train_model(config_file, parsed_filename):
    
    config = json.load(config_file)
    pdb.set_trace()

    # load data (using dataloader)
    # build vocab (check pa4) ??
    # build base model
    # train model based on config
    # save model into corresponding config_file name in models/config/trained