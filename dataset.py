#
# module_name, package_name, ClassName, method_name,
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME,
# global_var_name, instance_var_name, function_parameter_name,
# local_var_name.

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Da Real MVPs:
# https://github.com/deepsound-project/samplernn-pytorch
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import os
from os import listdir
from os.path import join
import numpy as np
from librosa.core import load
from natsort import natsorted
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm



def linear_quantize(samples, q_levels, epsilon=1e-2):
    # https://github.com/deepsound-project/samplernn-pytorch
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - epsilon
    samples += epsilon / 2
    return samples.long()


def linear_dequantize(samples, q_levels):
    # https://github.com/deepsound-project/samplernn-pytorch
    return samples.float() / (q_levels / 2) - 1


def q_zero(q_levels):
    # https://github.com/deepsound-project/samplernn-pytorch
    return q_levels // 2


class FolderDataset(Dataset):
    '''
    Reads all files in a folder and stores in a list.
    When __getitem__ is called the audiofile is loaded using librosa, quantized
    into q_levels and returned as torch.Tensor
    '''
    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels

        filePaths = [join(path, file_name) for file_name in listdir(path)]
        filePaths.sort()

        # Only use certain parts of total files
        # used in test/train, smaller dataset etc
        self.filePaths = filePaths[
            int(ratio_min * len(filePaths)) : int(ratio_max * len(filePaths))
        ]

    def __getitem__(self, index):
        (seq, _) = load(self.filePaths[index], sr=None, mono=True)
        return torch.cat([
            torch.LongTensor(self.overlap_len) \
                 .fill_(q_zero(self.q_levels)),
            linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ])

    def __len__(self):
        return len(self.filePaths)



class Dataloader(DataLoader):
    def __init__(self, dataset, batch_size, seq_len, overlap_len,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len

    def __iter__(self):
        # calls the regular dataloader __iter__ and gets quantized torch tensor
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()
            reset = True
            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
                from_index = seq_begin - self.overlap_len
                to_index = seq_begin + self.seq_len
                sequences = batch[:, from_index : to_index]
                input_sequences = sequences[:, : -1]
                target_sequences = sequences[:, self.overlap_len :].contiguous()
                yield (input_sequences, reset, target_sequences)
                reset = False

    def __len__(self):
        raise NotImplementedError()



def  makeDataLoader(path,
                    overlap_len,
                    q_levels,
                    ratio_min=0,
                    ratio_max=0.1,
                    batch_size=32,
                    seq_len=32):
    ds = FolderDataset(path, overlap_len, q_levels, ratio_min, ratio_max)
    dl = Dataloader(ds, batch_size, seq_len, overlap_len)
    return dl


