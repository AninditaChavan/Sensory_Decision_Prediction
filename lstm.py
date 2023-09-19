import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=hidden_size, batch_first=True)
        self.features_dense = nn.Linear(feature_size, 32)
        self.dense_1 = nn.Linear(hidden_size + 32, 32)
        self.output_layer = nn.Linear(32, 3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, time_series, features, lengths):
        # lstm part of netowork
        packed_time_series = pack_padded_sequence(time_series, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_out, (_, _) = self.lstm(packed_time_series) # get LSTM output
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True) # Unpack packed
        lstm_out = lstm_out[range(len(lstm_out)), lengths - 1, :] # Get last output
        # feed forward part of network
        features_out = torch.relu(self.features_dense(features))
        # merge lstm and features
        merged = torch.cat((lstm_out, features_out), dim=1)
        dense_out = torch.relu(self.dense_1(merged))
        # output layer
        output = self.output_layer(dense_out)
        #output = self.softmax(self.output_layer(dense_out))
        #print(output.shape)
        return output
