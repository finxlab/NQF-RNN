#!/usr/bin/env python
# coding: utf-8

# In[110]:

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import json
import shutil

logger = logging.getLogger('DeepAR Dataset Load') 
class Gradient(nn.Module):
    
    def forward(self, x: torch.Tensor, f : torch.Tensor) :
        gradients, *_ = torch.autograd.grad(outputs = f, inputs = x, grad_outputs = x.new_ones(x.shape), create_graph = True)

        return gradients
        
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        if len(x.size()) == 3 :
            y = y.view(x.size(0), x.size(1), y.size(-1)) 

        if len(x.size()) == 4:
            y = y.view(x.size(0), x.size(1), x.size(2), y.size(-1))  
            
        return y


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias= nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, input):
        return nn.functional.linear(input, torch.square(self.weight), self.bias)



class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params
        self.embedding = TimeDistributed(nn.Embedding(params["num_class"], params["embedding_dim"]))
        
        self.lstm = nn.LSTM(input_size= 1 + params["cov_dim"] + params["embedding_dim"],
                            hidden_size = params["lstm_hidden_dim"],
                            num_layers = params["lstm_layers"],
                            bias = True,
                            batch_first = False,
                            dropout = params["lstm_dropout"])
        
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        layers = []
        hidden_dim = params["layers"]

        layers.append(TimeDistributed(PositiveLinear(params["lstm_hidden_dim"] + 1, hidden_dim[0])))
        layers.append(TimeDistributed(nn.Tanh()))
        prev_h_dim = hidden_dim[0]
        for i in hidden_dim[1:-1] :
            layers.append(TimeDistributed(PositiveLinear(prev_h_dim, i)))
            prev_h_dim = i
            layers.append(TimeDistributed(nn.Tanh()))
        layers.append(TimeDistributed(PositiveLinear(prev_h_dim, 1)))
        # nn.ReLU?
        self.tanh = TimeDistributed(nn.Tanh())
        self.F_layer = nn.Sequential(*layers)


    def init_hidden(self, input_size):
        return torch.zeros(self.params["lstm_layers"], input_size, self.params["lstm_hidden_dim"], device=self.params["device"])

    def init_cell(self, input_size):
        return torch.zeros(self.params["lstm_layers"], input_size, self.params["lstm_hidden_dim"], device=self.params["device"])

    def forward(self, x, idx, hidden, cell):
        '''
        Args:
            x: ([T, batch_size, 1+cov_dim]): z_{t-1-T} + x_t-T ~ z_{t-1} + x_t (T timeseries), note that z_0 = 0
            idx ([T, batch_size]): one integer denoting the time series id (T timeseries)
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([T, batch_size]): estimated mean of z_t-T ~ z_t (T timeseries)
            sigma ([T, batch_size]): estimated standard deviation of z_t-T ~ z_t (T timeseries)
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        embedding_layer = self.embedding(idx)
        # print(idx.shape, embedding_layer.shape)
        lstm_input = torch.cat((x, embedding_layer), dim = 2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.tanh(output)
        return output, hidden, cell

    def F_inv(self, h, alpha):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-T
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-T
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        
        alpha = torch.unsqueeze(alpha, dim = -1)
        alpha = torch.cat((h, alpha), dim = -1)

        y_hat = self.F_layer(alpha)
        return torch.squeeze(y_hat)
    
    @torch.no_grad()
    def cdf(self, h, alpha, error = 1e-3) :
        '''
        Args:
            dist : ([T, batch_size]) : distribution of y
            alpha : ([N, T, batch_size])
        Returns:
            y : ([T, batch_size, 1])
            
        '''
        lowerbound = 0
        upperbound = 1
        T, batch_size = alpha.shape

        quantile_esti = torch.ones((2, T, batch_size)).to(self.params["device"])  
        quantile_esti[0] = quantile_esti[0] * lowerbound # lowerbound 0
        quantile_esti[1] = quantile_esti[1] * upperbound # upperbound 10
        

        for i in range(math.ceil(- np.log2(error/ (upperbound - lowerbound)))) :
    
            quantile_cand = (quantile_esti[1] + quantile_esti[0]) /2
            
            q = self.F(h, quantile_cand)
            quantile_esti[0] = torch.where(q <= alpha, quantile_cand, quantile_esti[0])
            quantile_esti[1] = torch.where(q >= alpha, quantile_cand, quantile_esti[1])
            
        quantile_cand = (quantile_esti[1] + quantile_esti[0])/2

        return quantile_cand

    
    @torch.no_grad()
    def cdf_test(self, h, alpha, error = 1e-3) :
        '''
        Args:
            dist : ([T, batch_size]) : distribution of y
            alpha : ([N, T, batch_size])
        Returns:
            y : ([T, batch_size, 1])
            
        '''
        lowerbound = 0
        upperbound = 1
         
        quantile_esti = torch.ones((2, alpha.shape[1])).to(self.params["device"])  
        quantile_esti[0] = quantile_esti[0] * lowerbound # lowerbound 0
        quantile_esti[1] = quantile_esti[1] * upperbound # upperbound 10
        

        for i in range(math.ceil(- np.log2(error/ (upperbound - lowerbound)))) :
    
            quantile_cand = (quantile_esti[1] + quantile_esti[0]) /2
            q = self.F(h, quantile_cand)
            quantile_esti[0] = torch.where(q <= alpha, quantile_cand, quantile_esti[0])
            quantile_esti[1] = torch.where(q > alpha, quantile_cand, quantile_esti[1])
            
        quantile_cand = (quantile_esti[1] + quantile_esti[0])/2

        return quantile_cand

    @torch.no_grad()
    def model_init(self, train_loader):
    
        self.eval()
        train_batch, idx, v, labels_batch = next(iter(train_loader))
        
        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(self.params["device"])  
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(self.params["device"])
        idx = idx.permute(1, 0, 2).to(self.params["device"])

        T, batch_size = labels_batch.shape

        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)
        h, _, _ = self(train_batch, idx, hidden, cell)

        h = h[-self.params['predict_steps']:]
        labels_batch = labels_batch[-self.params['predict_steps']:]
        
        _step = 0.001 
        alpha_array = torch.arange(start=0, end = 1, step=_step, device = self.params["device"]).view(-1, 1, 1).repeat(1, self.params['predict_steps'], batch_size)
        h = torch.unsqueeze(h, dim = 0).repeat(alpha_array.shape[0], 1, 1, 1)
        y = self.F_inv(h, alpha_array)
        
        pinball_loss = 2 * (alpha_array - (y > labels_batch).to(torch.float)) * (labels_batch - y) 
        pinball_loss = (pinball_loss[1:] + pinball_loss[:-1]) / 2 * _step

        loss = torch.sum(torch.sum(pinball_loss, dim = 0)) / self.params['predict_steps'] / T

        return loss

    def test(self, x, v_batch, id_batch, hidden, cell, sampling_n = 200) :
        
        '''
        Args:
            x: ([T, batch_size, 1+cov_dim]): z_0 + x_0 ~ z_{t-1} + x_t (T timeseries), note that z_0 = 0
            v_batch ([batch_size, 1])
            idx ([T, batch_size]): one integer denoting the time series id (T timeseries)
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step 0
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step 0
        
        Returns:
            samples ([sample_n, batch_size, predict_time]): sampling of z_t-predict_time ~ z_t (predict_time timeseries)
            sample_mu ([predict_time, batch_size]): sampling mu of z_t-predict_time ~ z_t (predict_time timeseries)
            sample_sigma ([predict_time, batch_size]): sampling sigma of z_t-predict_time ~ z_t (predict_time timeseries)
        '''
        with torch.no_grad():
            T, batch_size, _ = x.shape
            samples = torch.zeros(sampling_n, batch_size, self.params['predict_steps'], device=self.params['device'])

            x_repeat = x.repeat(1, sampling_n, 1)
            idx_repeat = id_batch.repeat(1, sampling_n, 1)
            hidden_repeat = hidden.repeat(1, sampling_n, 1)
            cell_repeat = cell.repeat(1, sampling_n, 1)

            decoder_hidden = torch.zeros_like(hidden_repeat, device=self.params['device'])
            decoder_cell = torch.zeros_like(cell_repeat, device=self.params['device'])
        
            loop_T = int(math.ceil(sampling_n / self.params['max_sampling_n'] ))

           # Memory issue

            for i in range(loop_T) :
                N_start, N_end = i * self.params['max_sampling_n'] * batch_size, (i + 1) * self.params['max_sampling_n'] * batch_size

                x_encoder = x_repeat[:self.params['predict_start'], N_start : N_end].contiguous()
                idx_encoder = idx_repeat[:self.params['predict_start'], N_start : N_end].contiguous()
                hidden_encoder = hidden_repeat[:, N_start : N_end].contiguous()
                cell_encoder = cell_repeat[:, N_start : N_end].contiguous()

                _, decoder_hidden_t, decoder_cell_t = self(x_encoder, idx_encoder, hidden_encoder, cell_encoder)

                decoder_hidden[:, N_start : N_end] = decoder_hidden_t
                decoder_cell[:, N_start : N_end] = decoder_cell_t



            decoder_hidden = decoder_hidden.contiguous()
            decoder_cell = decoder_cell.contiguous()
        
            for t in range(self.params['predict_steps']):
                t_pred = self.params['predict_start'] + t
                h_de, decoder_hidden, decoder_cell = self(x_repeat[t_pred : t_pred + 1], idx_repeat[t_pred : t_pred + 1], decoder_hidden, decoder_cell)

                uniform = torch.distributions.uniform.Uniform(0.01, 0.99).sample(h_de[:, :, 0].shape).to(h_de.device)
                pred = self.F_inv(h_de, uniform)
                samples[:, :, t] = pred.view(sampling_n, batch_size) * v_batch.reshape(1, -1)

                if t < (self.params['predict_steps'] - 1):
                    x_repeat[self.params['predict_start'] + t + 1, :, 0] = pred
            
            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            
        return samples, torch.permute(sample_mu, (1,0)).contiguous(), torch.permute(sample_sigma, (1,0)).contiguous()    
        
    


def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'ins_{ins_name}_best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')


def load_checkpoint(checkpoint, model, optimizer = None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
        
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
        
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
        optimizer.param_groups[0]['capturable'] = True

    return checkpoint

    
def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        # d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_json(path) :
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data


