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

        embed_size = config['model']['embedding_size']
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_layers']
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)


    # expects encoded tweets in 1D tensor form (up to 280 length)
    def forward(self, encoded_tweet, train = False):

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
                out_tweet = []

                pass

            # begin tweet with phrase
            else:

                start_len = encoded_tweet.shape[0]

                # feed in the first characters and get the lstm state
                embedded_tweet = self.embed(encoded_tweet)
                _, lstm_state = self.lstm()    
                out_tweet = encoded_tweet

            # generate tweet
            for i in range(self.timesteps - start_len):

                # TODO: FINSIH THIS WHEN i CAN DEBUG EASILY
                pass

