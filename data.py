import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import time
from datetime import timedelta
device = torch.device('cuda') 

# data loader
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device,shuffle=False):
        self.batch_size = batch_size
        self.batches =  batches
        self.n_batches = len(self.batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        self.shuffle=shuffle
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):

        x = torch.FloatTensor([_[0] for _ in datas]).to(self.device)
        
        y = torch.FloatTensor([_[1] for _ in datas]).to(self.device).reshape(-1)

        return [x,y]
        
        
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset,batch_size=128,shuffle=True):
    iter = DatasetIterater(dataset, batch_size,device,shuffle)
    return iter
