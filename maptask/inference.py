import matplotlib.pyplot as plt
from tqdm import tqdm

from maptask.dataset import DSet
from maptask.models import BasicLstm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from maptask.utils import plot_pitch_intensity

def plot(out, target):
    pitch = out[0].numpy()
    intensity = out[1].numpy()
    target_pitch = target[0].numpy()
    target_intensity = target[1].numpy()

    plt.subplot(2,1,1)
    plt.title('Pitch')
    plt.plot(pitch, 'b', label='Predicted')
    plt.plot(target_pitch, 'r', label='Target')
    plt.legend()
    plt.xlabel('frames')
    plt.ylabel('frequency', color='k')
    plt.tick_params('y', colors='k')
    plt.subplot(2,1,2)
    plt.title('Intensity')
    plt.plot(intensity, 'b', label='Predicted')
    plt.plot(target_intensity, 'r', label='Target')
    plt.legend()
    plt.ylabel('intensity', color='k')
    plt.tick_params('frames', colors='k')
    plt.tight_layout()
    plt.pause(0.1)


# Data
dset = DSet()
dloader = DataLoader(dset, batch_size=64, shuffle=True)

# Load model
model_path = 'checkpoints/basiclstm_l5_best.pt'

print('Loading model: ', model_path)
old_model = torch.load(model_path)
old_model.lstm.flatten_parameters()
state_dict = old_model.state_dict()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BasicLstm(rnn_layers=5, dropout=0.5).to(device)
model.load_state_dict(state_dict)
model.eval()

inp, target = dset.get_random()
inp = inp.unsqueeze(0).to(device)
out = model.inference(inp)
print('Final out shape: ', out.shape)
plot(out.cpu().detach().squeeze(0), target)

print('Iterate through data set comparing predictions with target: ')
while True:
    inp, target = dset.get_random()
    inp = inp.unsqueeze(0).to(device)
    out = model(inp).squeeze()
    plot(out.cpu().detach(), target)
    ans = input('Break? (y/n)\n> ')
    if ans == 'y':
        plt.close()
        break
    plt.close()
