#!/usr/bin/env python
# coding: utf-8

# In[14]:


from __future__ import division
import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger('DeepAR Dataset Load')

class TrainDataset(Dataset):
    def __init__(self, data_path, data_name, num_class):
        self.data = np.load(os.path.join(data_path, f'traindata_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'trainv_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'trainlabel_{data_name}.npy'))
        self.train_len = self.data.shape[0]
        logger.info(f'train_len: {self.train_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        # final index => item number
        return (self.data[index, : , :-1],int(self.data[index, 0, -1]), self.label[index])
    
class TrainValDataset(Dataset):
    def __init__(self, data_path, data_name, num_class, random_seed = 0, datatype = 'train'):
        self.data = np.load(os.path.join(data_path, f'traindata_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'trainv_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'trainlabel_{data_name}.npy'))
        self.data_len = self.data.shape[0]
        self.datatype = datatype
        
        index_array = np.random.RandomState(seed=random_seed).permutation(self.data_len)
        train_index = sorted(index_array[:int(self.data_len * 0.9)])
        val_index = sorted(index_array[int(self.data_len * 0.9):])
        self.train_len = len(train_index)
        self.val_len = len(val_index)
        
        self.train_data = self.data[train_index]
        self.train_v = self.v[train_index]
        self.train_label = self.label[train_index]
        
        self.val_data = self.data[val_index]
        self.val_v = self.v[val_index]
        self.val_label = self.label[val_index]
        
        logger.info(f'train_len: {self.train_len}')
        logger.info(f'val_len: {self.val_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        if self.datatype == 'train':
            return self.train_len
        if self.datatype == 'validation':
            return self.val_len

    def __getitem__(self, index):
        if self.datatype == 'train':
            return (self.train_data[index, : , :-1],self.train_data[index, :, -1:].astype(int), self.train_v[index], self.train_label[index])
        
        if self.datatype == 'validation':
            return (self.val_data[index,:,:-1],self.val_data[index, :, -1:].astype(int), self.val_v[index], self.val_label[index])


class TestDataset(Dataset):
    def __init__(self, data_path, data_name, num_class):
        self.data = np.load(os.path.join(data_path, f'testdata_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'testv_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'testlabel_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        logger.info(f'test_len: {self.test_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],self.data[index,:,-1:].astype(int), self.v[index], self.label[index])

    
class WeightedSampler(Sampler):
    def __init__(self, data_path, data_name, random_seed = 0, replacement=True):

        v = np.load(os.path.join(data_path, f'trainv_{data_name}.npy'))
        self.data_len = v.shape[0]
        
        index_array = np.random.RandomState(seed=random_seed).permutation(self.data_len)
        train_index = sorted(index_array[:int(self.data_len * 0.9)])
        
        v = v[train_index]
        self.weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
        logger.info(f'weights: {self.weights}')
        
        self.num_samples = self.weights.shape[0]
        logger.info(f'num samples: {self.num_samples}')
        
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

