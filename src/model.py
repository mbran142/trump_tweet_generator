import torch
import torch.nn as nn
import torchvision.models as models
import pdb

# model to generate tweets
class tweet_model(nn.Module):

    def __init__(self, config):
        
        super(tweet_model, self).__init__()

        # load config
        self.timesteps = 290  # max tweet length is 280, add 10 due to '&' -> 'and' conversion
        self.vocab_size = 32
        self.temperature = 0.2

        embed_size = config['model']['embedding_size']
        num_lstm_layers = config['model']['num_lstm_layers']
        lstm_size = config['model']['lstm_size']

        hidden_size = []
        for i in range(4):
            hidden_size.append(config['model'][f'hidden_size_{i + 1}'])

        # model looks something like this
        #
        #                   out 
        #                    |
        #                  final
        #                    |
        #                   fc4
        #                  /   \
        #                 |    fc3
        #               lstm    |
        #                 |    fc2
        #                  \   /
        #                   fc1
        #                    |
        #                  embed
        #                    |
        #                  input

        self.embed = nn.Embedding(self.vocab_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, hidden_size[0])

        self.lstm = nn.LSTM(hidden_size[0], lstm_size, num_lstm_layers, batch_first = True)

        self.fc_2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc_3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.fc_4 = nn.Linear(lstm_size + hidden_size[2], hidden_size[3])
        self.fc_final = nn.Linear(hidden_size[3], self.vocab_size)

        # initialize fc weights
        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)
        nn.init.xavier_uniform_(self.fc_3.weight)
        nn.init.xavier_uniform_(self.fc_4.weight)
        nn.init.xavier_uniform_(self.fc_final.weight)


    # modify model temperature (default is 0.2)
    def change_temperature(self, new_temp):
        self.temperature = new_temp


    # choose a character from the distribution the model produces
    def pick_char(self, distribution):
        
        sm = nn.Softmax(dim = 1)
        distribution = sm(distribution / self.temperature)

        sample = torch.multinomial(torch.squeeze(distribution, 1), num_samples = 1)
        return sample


    # expects encoded tweets in 1D tensor form (up to 290 length)
    def forward(self, encoded_tweet = None, train = False):

        # train via teacher forcing
        if train:
            return self.propagate(encoded_tweet)[0]     # ignore lstm state output

        # generate tweet using model
        else:

            start_len = encoded_tweet.shape[0]

            # feed in the tweet start and get the lstm state
            char_distrib, lstm_state = self.propagate(encoded_tweet)

            # get last character from distribution
            char_distrib = torch.squeeze(char_distrib, 0)[-1:]

            next_char = self.pick_char(char_distrib)
            out_tweet = torch.cat(encoded_tweet, torch.tensor(next_char)) 

            # generate rest of tweet
            for i in range(self.timesteps - start_len):

                char_distrib, lstm_state = self.propagate(next_char, lstm_state)

                next_char = self.pick_char(char_distrib)
                out_tweet = torch.cat((out_tweet, torch.squeeze(next_char, 0)), 0)

            return out_tweet

    
    # propagate input through model
    def propagate(self, input, lstm_state = None):

        embedded_tweet = self.embed(input)
        fc1_out = self.fc_1(embedded_tweet)

        lstm_out, lstm_state = self.lstm(fc1_out, lstm_state)

        fc2_out = self.fc_2(fc1_out)
        fc3_out = self.fc_3(fc2_out)

        fc4_out = self.fc_4(torch.cat((lstm_out, fc3_out), 2))

        final_out = self.fc_final(fc4_out)

        return final_out, lstm_state