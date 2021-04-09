import torch
import torch.nn as nn
import torchvision.models as models
import pdb

# model to generate tweets
class tweet_model(nn.Module):

    def __init__(self, config):
        
        super(tweet_model, self).__init__()

        self.timesteps = 290  # max tweet length is 280, add 10 due to '&' -> 'and' conversion
        self.vocab_size = 32
        self.temperature = 0.2

        embed_size = config['model']['embedding_size']
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_layers']

        self.embed = nn.Embedding(self.vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)

        nn.init.xavier_uniform_(self.fc.weight)


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
            
            embedded_tweet = self.embed(encoded_tweet)
            lstm_out, _ = self.lstm(embedded_tweet)
            fc_out = self.fc(lstm_out)

            return fc_out

        # generate tweet using model
        else:

            start_len = encoded_tweet.shape[0]

            # feed in the tweet start and get the lstm state
            embedded_tweet = self.embed(encoded_tweet)
            lstm_out, lstm_state = self.lstm(torch.unsqueeze(embedded_tweet, 0))    
            char_distrib = self.fc(lstm_out)

            # get last character from distribution
            char_distrib = torch.squeeze(char_distrib, 0)[-1:]

            next_char = self.pick_char(char_distrib)
            out_tweet = torch.cat(encoded_tweet, torch.tensor(next_char)) 

            # generate rest of tweet
            for i in range(self.timesteps - start_len):
                
                next_char = self.pick_char(char_distrib)
                out_tweet = torch.cat((out_tweet, torch.squeeze(next_char, 0)), 0)

                embed_out = self.embed(next_char)
                lstm_out, lstm_state = self.lstm(embed_out, lstm_state)
                char_distrib = self.fc(lstm_out)

            return out_tweet
