from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

def plot_pitch_intensity(pitch, intensity, title='Pitch & intensity'):
    if not isinstance(pitch, np.ndarray):
        pitch = pitch.numpy()
    if not isinstance(intensity, np.ndarray):
        intensity = intensity.numpy()
    fig, ax1 = plt.subplots()
    plt.title(title)
    ax1.plot(pitch, 'b')
    ax1.set_xlabel('frames')
    ax1.set_ylabel('frequency', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(intensity, 'y')
    ax2.set_ylabel('intensity', color='y')
    ax2.tick_params('frames', colors='y')
    fig.tight_layout()
    plt.show()


class DSet(Dataset):
    def __init__(self, root_dir='data'):
        self.root_dir = root_dir
        self.pitch = torch.tensor(np.load(join(root_dir, 'pitch.npy'))[:, :,
                                                                       :-1]).float()
        self.intensity = torch.tensor(np.load(join(root_dir,
                                                   'intensity.npy'))).float()

    def __len__(self):
        return self.pitch.shape[1]

    def __getitem__(self, idx):
        data = torch.stack((self.pitch[0, idx, :-1],
                           self.intensity[0, idx, :-1],
                           self.pitch[1, idx, :-1],
                           self.intensity[1, idx, :-1]))
        target = torch.stack((self.pitch[1, idx, 1:],
                              self.intensity[1, idx, 1:]))
        return data, target


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

    def old_forward(self, inputs, hidden=None):
        batch_size = inputs.shape[0]
        n_data = inputs.shape[1]
        n_frames = inputs.shape[2]

        if not hidden:
            (h, c) = self.init_hidden(batch_size)
            h, c = h.to(inputs.device), c.to(inputs.device)

        all_out = []
        for i in range(n_frames):
            if i % self.bptt_length:
                h = h.detach()
                c = c.detach()
            x = inputs[:, :, i]
            x = x.unsqueeze(1)
            x, (h, c)= self.lstm(x, (h, c))
            out = self.output(x)
            out = out.permute(0,2,1)
            all_out.append(out)

        out = torch.stack(all_out, dim=1)  # -> (Batch, prediction, 1) stack list of tensors
        out = out.squeeze()  # -> (Batch, prediction)
        out = out.permute(0, 2, 1)
        return out

    def training(self, optimizer, loss_fn, dloader):
        print('Size of data: ', len(dloader))


# Data
dset = DSet()
dloader = DataLoader(dset, batch_size=64, shuffle=True)


# Tensorboard
use_tensorboard = True
if use_tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
else:
    data, targets = dset[np.random.randint(len(dset))]
    plot_pitch_intensity(data[0], data[1])


for rnn_layers in [1, 2, 5, 10, 15, 20]:
    # Model
    n_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BasicLstm(rnn_layers=rnn_layers, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model

    model.train()
    epoch_loss = []
    i = 0
    for epoch in range(1, n_epochs+1):
        batch_loss = []
        for inputs, targets in tqdm(dloader, desc='Epoch: {}/{}'.format(epoch, n_epochs)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = model(inputs)
            # print('out shape: ', out.shape)
            # print('targets shape: ', targets.shape)
            # input()
            optimizer.zero_grad()
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
            if use_tensorboard:
                writer.add_scalar('Batch loss ({} layers)'.format(rnn_layers), loss.item(), i)
            batch_loss.append(loss.item())
            i += 1
        tmp_epoch_loss = torch.tensor(batch_loss).mean().item()
        epoch_loss.append(tmp_epoch_loss)
        if use_tensorboard:
            writer.add_scalar('Epoch loss ({} layers)'.format(rnn_layers), tmp_epoch_loss, epoch)
        else:
            plt.plot(epoch_loss)
            plt.pause(0.1)

    torch.save(model, 'checkpoints/basiclstm_{}_layers.pt'.format(rnn_layers))
