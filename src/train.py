from datasets import tweet_dataset, build_tweet_datasets
from model import tweet_model
import json
import numpy as np
import pdb
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import traceback

def build_and_train_model(config_file, parsed_filename):
    
    try:
        with open('models/config/' + config_file) as f:
            config = json.load(f)
        if 'training' not in config or 'model' not in config:
            raise Exception()
        if 'epochs' not in config['training'] or 'learning_rate' not in config['training'] or 'patience' not in config['training'] or 'batch_size' not in config['training']:
            raise Exception()
        if 'hidden_size' not in config['model'] or 'embedding_size' not in config['model'] or 'num_layers' not in config['model']:
            raise Exception()
    except:
        print(f'Error: {config_file} not in proper format')
        return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using \'' + device + '\' for training...')

    model = tweet_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['training']['learning_rate']) 
    criterion = torch.nn.CrossEntropyLoss()

    train_set, val_set = build_tweet_datasets('data/' + parsed_filename)
    train_dataloader = DataLoader(train_set, shuffle = True, batch_size = config['training']['batch_size'])
    val_dataloader = DataLoader(val_set, shuffle = True, batch_size = config['training']['batch_size'])

    if torch.cuda.is_available():
        model = model.cuda().float()
        criterion = criterion.cuda()

    # train model
    try:
    
        violation = 0
        best_loss = 1_000_000      # arbitrary large number

        train_losses = []
        val_losses = []

        for epoch in range(config['training']['epochs']):

            train_losses.append(get_loss(model, train_dataloader, device, optimizer, criterion, config['training']['batch_size']))                 # train model
            val_losses.append(get_loss(model, val_dataloader, device, optimizer, criterion, config['training']['batch_size'], train = False))      # validation

            print(f'Epoch {epoch + 1}: training loss = {train_losses[-1]} | validation loss = {val_losses[-1]}')

            # consider early stopping
            if epoch <= 1 and val_losses[-1] > val_losses[-2]:

                violation += 1

                # stop early
                if violation == config['training']['patience']:

                    print(f'Stopping early at epoch {epoch + 1}.')
                    break

            # best model so far    
            elif val_losses[-1] < best_loss:

                best_model = deepcopy(model)
                best_epoch = epoch + 1

        print(f'Saving model from {best_epoch} into \'models/trained/{config_file[ : -5]}\'')

        torch.save(best_model, 'models/trained/' + config_file[ : -5])

    except Exception as e:
        print('--- ERROR DURING TRAINING ---')
        print(e)
        traceback.print_exc()
        return False

    return True


# get loss value on input dataset. Used for both training and validation
def get_loss(model, dataset, device, optimizer, criterion, batch_size, train = True):

    num_iterations = len(dataset) // batch_size
    print('Training:' if train else 'Validation:', f'{num_iterations} iterations.')
    print('.' * num_iterations)

    loss_val = 0

    for i, tweet in enumerate(dataset):

        if train:
            optimizer.zero_grad()

        tweet = tweet.to(device)

        outputs = model(tweet, train = True)
        loss = criterion(torch.flatten(outputs, 0, 1), torch.flatten(tweet))
        loss_val += loss.item()
        
        if train:
            loss.backward()
            optimizer.step()

        print('.')

    loss_val = loss_val / (len(dataset) / batch_size)
    return loss_val


# convert tweet ascii into 1D tensor
def tweet_to_tensor(tweet):

    return torch.Te