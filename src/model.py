import torch
import torch.nn as nn
import torchvision.models as models
import pdb

# model to generate tweets
class tweet_model(nn.Module):

    def __init__(self, config):
        
        super(tweet_model, self).__init__()

        self.timesteps = 280  # max tweet length is 280
        self.vocab_size = 129 # 0-127 ASCII, 128 = <pad>
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


    # expects encoded tweets in 1D tensor form (up to 280 length)
    def forward(self, encoded_tweet = None, train = False):

        # train via teacher forcing
        if train:
            
            embedded_tweet = self.embed(encoded_tweet)
            lstm_out, _ = self.lstm(embedded_tweet)
            fc_out = self.fc(lstm_out)

            return fc_out

        # utilize trained model
        else:

            # completely new tweet
            if encoded_tweet is None:

                start_len = 0
                lstm_state = None
                out_tweet = torch.IntTensor([])

                # randomly choose an alphabetical character to start with
                char_distrib = torch.tensor(([0] * 64) + ([1] * 26) + ([0] * 6) + ([1] * 26) + ([0] * 7))
                char_distrib = torch.unsqueeze(char_distrib, 0)

            # begin tweet with phrase
            else:

                start_len = encoded_tweet.shape[0]

                # feed in the first characters and get the lstm state
                embedded_tweet = self.embed(encoded_tweet)
                lstm_out, lstm_state = self.lstm(torch.unsqueeze(embedded_tweet, 0))    
                char_distrib = self.fc(lstm_out)

                # last character from distribution
                char_distrib = torch.squeeze(char_distrib, 0)[-1:]

                out_tweet = encoded_tweet

            # generate tweet
            for i in range(self.timesteps - start_len):
                
                next_char = self.pick_char(char_distrib)
                out_tweet = torch.cat((out_tweet, torch.squeeze(next_char, 0)), 0)

                embed_out = self.embed(next_char)
                lstm_out, lstm_state = self.lstm(embed_out, lstm_state)
                char_distrib = self.fc(lstm_out)

            return out_tweet
