from __future__ import division
import torch.utils.data
import torch
import numpy as np 
import os
import time
import pandas as pd
import pdb

def make_dataset(data_dir, usage):
    if usage not in ['test', 'eval', 'train']:
        raise ValueError('usage must be test train or eval')

    if usage in ['train', 'eval']:
        datas = []
        for rt,dirs,files in os.walk(data_dir):
            for file in files:
                datas.append(os.path.join(rt, file))
        len_datas = len(datas)
        if usage == 'train':
            return datas[len_datas//5:]
        else:
            return datas[:len_datas//5]
    else:
        return open(data_dir, 'r').readlines()

class data(torch.utils.data.Dataset):
    def __init__(self, data_dir, usage):        
        self.data_dir = data_dir
        self.usage = usage
        self.datas = make_dataset(data_dir, usage)

    def __getitem__(self, index):
        # start_time = time.time()
        if self.usage in ['train', 'eval']:
            fp = open(self.datas[index], 'r')
            data = list(map(float, fp.readline()[:-1].split(',')))
            # print(self.datas[index], time.time()-start_time)
            c = np.asarray(data[:2200])
            p = np.asarray(data[2200:3600])
            
            label = np.asarray([data[-1]])
            # pdb.set_trace()
            return torch.from_numpy(c).float(), \
                torch.from_numpy(p).float(), \
                torch.from_numpy(label).float()
        else:
            data = list(map(float, self.datas[index][:-1].split(',')))
            c = np.asarray(data[:2200])
            p = np.asarray(data[2200:3600])
            
            return torch.from_numpy(c).float(), \
                torch.from_numpy(p).float()

    def __len__(self):
        return len(self.datas)
