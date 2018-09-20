import numpy as np
from tqdm import tqdm

from maptask.dataset import DSet
from maptask.models import BasicLstm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


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


# Simple training loop for training a basic lstm model
# in -> lstm -> fc-layer -> out
for rnn_layers in [2, 5, 10, 15, 20]:
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
