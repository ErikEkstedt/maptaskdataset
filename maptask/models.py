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
                 teacher=0.5,
                 batch_first=True):
        super(BasicLstm, self).__init__()

        self.input_size = input_size
        self.hidden = hidden
        self.output_size = output_size
        self.rnn_layers = rnn_layers
        self.bptt_length = bptt_length
        self.teacher = teacher  # percentage to use teacher forcing
        self.batch_first = batch_first
        # Batch first = False:
        # The input dimensions are (seq_len, batch, input_size)
        # Batch first = True:
        # The input dimensions are (batch, seq_len, input_size)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden,
                            num_layers=rnn_layers,
                            batch_first=batch_first,
                            dropout=dropout)

        self.output = nn.Linear(hidden, output_size)

    def init_hidden(self, batch_size):
        '''(hidden, cell-state) initial values '''
        return (torch.zeros(self.rnn_layers, batch_size, self.hidden),
                torch.zeros(self.rnn_layers, batch_size, self.hidden))

    def inference(self, inputs, hidden=None, verbose=False):
        batch_size = inputs.shape[0]
        n_frames = inputs.shape[1]
        n_data = inputs.shape[2]
        if not hidden:
            (h, c) = self.init_hidden(batch_size)
            h, c = h.to(inputs.device), c.to(inputs.device)


        # inputs: (batch, sequence_len, elements)
        context = inputs[:, :, :2]  # the context data (no bc)
        # print('context.shape', context.shape)

        x0 = inputs[:, 0, :].unsqueeze(1)  # first frame input contains context and bc
        # print('x0.shape', x0.shape)

        # out0 = x0[:,:,-2:]
        # print('out0.shape', out0.shape)
        # all_out.append(out0)

        all_out = []  # store final output sequence

        out_lstm, (h,c) = self.lstm(x0, (h,c))  # initial state
        out = self.output(out_lstm.reshape(batch_size, 1, self.hidden))  # only one frame
        # print('0 out.shape', out.shape)
        all_out.append(out.squeeze(0))

        # Autoregressive
        for f in range(1, n_frames): # from t=1 to end of sequence
            context_frame = context[:, f]
            context_frame = context_frame.unsqueeze(1)
            # print('context_frame.shape', context_frame.shape)
            x = torch.cat((context_frame, out), dim=2)  # stack out and context
            # print('x.shape', x.shape)
            out_lstm, (h,c) = self.lstm(x, (h,c))  # initial state
            out = self.output(out_lstm.reshape(batch_size, 1, self.hidden))  # only one frame
            # print('out.shape', out.shape)
            all_out.append(out)
        return torch.cat(all_out, dim=1)

    def forward(self, inputs, hidden=None):
        # inputs: (batch, sequence_len, elements)
        # if np.random.rand() < self.teacher:
        if False:
            # teacher forcing
            batch_size = inputs.shape[0]
            n_frames = inputs.shape[1]
            n_data = inputs.shape[2]
            if not hidden:
                (h, c) = self.init_hidden(batch_size)
                h, c = h.to(inputs.device), c.to(inputs.device)
            out_lstm, (h, c) = self.lstm(inputs, (h,c)) # (batch, n_frames, hidden)
            out = self.output(out_lstm.reshape(batch_size*n_frames, -1))
            out = out.reshape(batch_size, n_frames, 2)
        else:
            out = self.inference(inputs)
        return out
