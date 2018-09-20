import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BasicLstm(nn.Module):
    def __init__(self,
                 input_size=4,
                 output_size=2,
                 hidden=64,
                 rnn_layers=2,
                 dropout=0.5,
                 bptt_length=16,
                 batch_first=True):
        super(BasicLstm, self).__init__()

        self.input_size = input_size
        self.hidden = hidden
        self.output_size = output_size
        self.rnn_layers = rnn_layers
        self.bptt_length = bptt_length

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden,
                            num_layers=rnn_layers,
                            batch_first=batch_first,
                            dropout=dropout)

        # Batch first = False:
        # The input dimensions are (seq_len, batch, input_size)
        # Batch first = True:
        # The input dimensions are (batch, seq_len, input_size)
        self.output = nn.Linear(hidden, output_size)

    def init_hidden(self, batch_size):
        '''(hidden, cell-state) initial values '''
        return (torch.zeros(self.rnn_layers, batch_size, self.hidden),
                torch.zeros(self.rnn_layers, batch_size, self.hidden))

    def forward(self, inputs, hidden=None):
        batch_size = inputs.shape[0]
        n_data = inputs.shape[1]
        n_frames = inputs.shape[2]
        # (batch, elements, sequence_len)
        x = inputs.permute(0, 2, 1) # -> (batch, sequence_len, elements)
        if not hidden:
            (h, c) = self.init_hidden(batch_size)
            h, c = h.to(inputs.device), c.to(inputs.device)
        out_lstm, (h, c) = self.lstm(x, (h,c)) # (batch, n_frames, hidden)
        # out = self.output(out_lstm.view(batch_size*n_frames, -1))  # contigous
        # call before view needed
        out = self.output(out_lstm.reshape(batch_size*n_frames, -1))
        return out.reshape(batch_size, 2, n_frames)
