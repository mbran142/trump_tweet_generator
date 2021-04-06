import torch
import json
from model import tweet_model
import pdb

# generate tweets using trained model
def use_model(model_filename):

    model, opt = load_model(model_filename)
    print(f'\'{model_filename}\' loaded successfully. Select action (default temperature is 0.2):')

    temp = 0.2
    done = False
    print_selection = True

    # main loop
    while not done:

        if print_selection:
            print('[1] Generate tweet from scratch.')
            print('[2] Generate tweet given starting input.')
            print(f'[3] Set temperature (current = {temp}).')
            print('[0] Exit.')

        print('Enter selection: ', end='')

        print_selection = True
        selection = input()
        
        pdb.set_trace()
        
        if len(selection) != 1:
            print('Invalid selection.')
            print_selection = False

        elif selection == '0':
            done = True

        elif selection == '1':
            tweet = generate_tweet(model)
            print(f'Tweet generated: {tweet}')

        elif selection == '2':
            print('Enter starting input: ', end = '')
            tweet_start = input()
            tweet = generate_tweet(model, start = tweet_start)
            print(f'Tweet generated: {tweet}')

        elif selection == '3':
            print('Enter new temperature: ', end = '')
            new_temp = input()
            temp = float(new_temp)
            model.change_temperature(temp)

        else:
            print('Invalid selection.')
            print_selection = False

    print('Program exiting.')


# generate tweet using model. Can use start phrase
def generate_tweet(model, start = ''):

    # generate tweet
    with torch.no_grad():

        # completely new tweet
        if start == '':
            model_output = model()

        # use start
        else:

            # convert to tensor format
            model_input = [ord(char) for char in start]
            model_input = torch.tensor(model_input)
            
            model_output = model(model_input)

    # convert output to string
    tweet_out = ''
    for char in model_output:
        if char == 128:
            break
        tweet_out += chr(char)

    return tweet_out


# load model
def load_model(filename):

    # load config file
    with open('models/config/' + filename + '.json') as f:
        config = json.load(f)

    # set up base nodel and optimizers
    model = tweet_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['training']['learning_rate']) 

    # load model
    state_dict = torch.load('models/trained/' + filename + '.pt')
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])

    return model, optimizer