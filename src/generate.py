import torch
import json
from model import tweet_model

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
            print('[4] Use file to generate tweets.')
            print('[0] Exit.')

        print('Enter selection: ', end='')

        print_selection = True
        selection = input()

        if len(selection) != 1:
            print('Invalid selection.')
            print_selection = False

        elif selection == '0':
            done = True

        elif selection == '1':
            pass

        elif selection == '2':
            pass

        elif selection == '3':
            pass

        elif selection == '4':
            pass

    print('Program exiting.')


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