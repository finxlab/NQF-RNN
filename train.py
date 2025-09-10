#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
import os

import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
from lib.datasetLoader import *
from lib.Model import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import easydict
from copy import deepcopy



# In[2]:

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
logger = logging.getLogger('NQF-RNN.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data_folder', default='dataset\\data', help='Parent dir of the dataset')
parser.add_argument('--model_name', default='model', help='Directory containing params.json')
parser.add_argument('--params_dir', default='params', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def accuracy_ND(mu, labels):
    # zero_index = (labels != 0)
    diff = torch.sum(torch.abs(mu - labels)).item()
    summation = torch.sum(torch.abs(labels)).item()
    
    return diff, summation


def accuracy_RMSE(mu, labels):
    T, N = labels.shape
    diff = torch.sum(torch.mul((mu - labels), (mu - labels))).item()
    summation = torch.sum(torch.abs(labels)).item()
    
    return diff, summation, N * T

def accuracy_NLL(mu, sigma, labels):
    T, N = mu.shape
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(labels)
    return - torch.sum(likelihood).item(), N * T


def accuracy_CRPS(prediction_samples, labels):
    ensemble, indices = torch.sort(prediction_samples, axis = 0)
    ensemble = ensemble.contiguous()
    true = torch.permute(torch.unsqueeze(labels, dim = 0), (0, 2, 1)).contiguous()
    T, N = labels.shape
    zero_index = torch.permute((labels != 0), (1,0))

    n_sample = ensemble.shape[0]
    alpha = 1 / (n_sample - 1)
    alpha_array = (torch.arange(n_sample) * alpha).to(device = true.device)
    alpha_array = torch.unsqueeze(torch.unsqueeze(alpha_array, dim = 1), dim = 2)
    pinball_loss = 2 * (alpha_array - (ensemble > true).to(torch.float)) * (true - ensemble) * alpha
    return torch.sum(torch.sum(pinball_loss, dim = 0)).item(), N * T

def accuracy_QLOSS(prediction_samples, labels, q = 0.5):
    ensemble, indices = torch.sort(prediction_samples, axis = 0)
    ensemble = ensemble.contiguous()
    quantile = torch.permute(torch.squeeze(torch.quantile(ensemble, q, dim = 0)), (1,0))
    T, N = labels.shape
    # print(labels.shape)
    # print(quantile.shape)

    qloss = (labels - quantile) * (q - (labels < quantile).to(torch.float))

    return torch.sum(qloss).item(), N*T

def accuracy_ECRPS(prediction_samples, labels):
    true = torch.permute(torch.unsqueeze(labels, dim = 0), (0, 2, 1))
    M, N, T = prediction_samples.shape
    
    

    qi = torch.unsqueeze(prediction_samples, dim = 0)
    qj = torch.unsqueeze(prediction_samples, dim = 1)
    # print((qi - qj).shape)
    # print((torch.sum(torch.abs(prediction_samples - true), dim = 0) / M).shape)
    # print(torch.sum(torch.sum(torch.abs(qi - qj), dim = 1), dim = 0).shape)
    
    upper = torch.permute(torch.sum(torch.abs(prediction_samples - true), dim = 0) / M - torch.sum(torch.sum(torch.abs(qi - qj), dim = 1), dim = 0) / (2 * M ** 2) , (1,0))
    # print(upper.shape)
    return torch.sum(upper).item(), N*T



def evaluate(model, test_loader, params, epoch, loss_type = ['ND']):
    model.eval()
    with torch.no_grad():
        summary_metric = {}
        eval_batch = {}
        
        for loss in loss_type :
            if loss == 'ND' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'RMSE' :
                eval_batch[loss] = np.zeros(3)
            elif loss == 'NLL' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'CRPS' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q10' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q50' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q90' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'ECRPS' :
                eval_batch[loss] = np.zeros(2)
            else :
                eval_batch[loss] = None
        
        
        
        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params["device"])  
            labels_batch = labels.permute(1, 0).to(torch.float32).to(params["device"])
            v_batch = v.to(torch.float32).to(params["device"])
            id_batch = id_batch.permute(1, 0, 2).to(params["device"])


            T, N = labels_batch.shape

            hidden = model.init_hidden(N)
            cell = model.init_cell(N)
            
            # ND, RMSE evaluation
            samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling_n = params['sample_times'])
             
            # NLL, CRPS evaluation
            true_labels = labels_batch[params['predict_start']:, :] * torch.permute(v_batch, (1,0))
            for loss in loss_type :
                if loss == 'ND' :
                    diff, summation = accuracy_ND(sample_mu, true_labels)
                    eval_batch[loss][0] += diff
                    eval_batch[loss][1] += summation
                    
                elif loss == 'RMSE' :
                    diff, summation, index_n = accuracy_RMSE(sample_mu, true_labels)
                    eval_batch[loss][0] += diff
                    eval_batch[loss][1] += summation
                    eval_batch[loss][2] += index_n
                    
                elif loss == 'NLL' :
                    NLL, index_n = accuracy_NLL(sample_mu, sample_sigma, true_labels)
                    eval_batch[loss][0] += NLL
                    eval_batch[loss][1] += index_n
                
                elif loss == 'CRPS' :
                    CRPS, index_n = accuracy_CRPS(samples, true_labels)
                    eval_batch[loss][0] += CRPS
                    eval_batch[loss][1] += index_n

                elif loss == 'Q10' :
                    upper, lower = accuracy_QLOSS(samples, true_labels, q = 0.1)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'Q50' :
                    upper, lower = accuracy_QLOSS(samples, true_labels, q = 0.5)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'Q90' :
                    upper, lower = accuracy_QLOSS(samples, true_labels, q = 0.9)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'ECRPS' :
                    upper, lower = accuracy_ECRPS(samples, true_labels)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower
                
                else :
                    eval_batch[loss] = None
                    
        for loss in loss_type :
            if loss == 'ND' :
                summary_metric[loss] =  eval_batch[loss][0]/ eval_batch[loss][1]
            elif loss == 'RMSE' :
                summary_metric[loss] = ((eval_batch[loss][0] / eval_batch[loss][2])**(1/2)) / (eval_batch[loss][1] / eval_batch[loss][2])
            elif loss == 'NLL' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'CRPS' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q10' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q50' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q90' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'ECRPS' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            else :
                summary_metric[loss] = None
                
    return summary_metric
def evaluate_test(model, test_loader, params, loss_type = ['ND']):
    
    model.eval()
    with torch.no_grad():
        summary_metric = {}
        eval_batch = {}
        
        for loss in loss_type :
            if loss == 'ND' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'RMSE' :
                eval_batch[loss] = np.zeros(3)
            elif loss == 'NLL' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'CRPS' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q10' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q50' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q90' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'ECRPS' :
                eval_batch[loss] = np.zeros(2)
            else :
                eval_batch[loss] = None
        
        
        
        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params["device"])  
            labels_batch = labels.permute(1, 0).to(torch.float32).to(params["device"])
            v_batch = v.to(torch.float32).to(params["device"])
            id_batch = id_batch.permute(1, 0, 2).to(params["device"])

            T, N = labels_batch.shape

            hidden = model.init_hidden(N)
            cell = model.init_cell(N)
            
            # print(v_batch)
            samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling_n = params['sample_times'])
            for loss in loss_type :
                if loss == 'ND' :
                    diff, summation = accuracy_ND(sample_mu, labels_batch[params['predict_start']:])
                    eval_batch[loss][0] += diff
                    eval_batch[loss][1] += summation
                    
                elif loss == 'RMSE' :
                    diff, summation, index_n = accuracy_RMSE(sample_mu, labels_batch[params['predict_start']:])
                    eval_batch[loss][0] += diff
                    eval_batch[loss][1] += summation
                    eval_batch[loss][2] += index_n
                    
                elif loss == 'NLL' :
                    NLL, index_n = accuracy_NLL(sample_mu, sample_sigma, labels_batch[params['predict_start']:])
                    eval_batch[loss][0] += NLL
                    eval_batch[loss][1] += index_n
                
                elif loss == 'CRPS' :
                    CRPS, index_n = accuracy_CRPS(samples, labels_batch[params['predict_start']:])
                    eval_batch[loss][0] += CRPS
                    eval_batch[loss][1] += index_n

                elif loss == 'Q10' :
                    upper, lower = accuracy_QLOSS(samples, labels_batch[params['predict_start']:], q = 0.1)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'Q50' :
                    upper, lower = accuracy_QLOSS(samples, labels_batch[params['predict_start']:], q = 0.5)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'Q90' :
                    upper, lower = accuracy_QLOSS(samples, labels_batch[params['predict_start']:], q = 0.9)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'ECRPS' :
                    upper, lower = accuracy_ECRPS(samples, labels_batch[params['predict_start']:])
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower
                
                
                else :
                    eval_batch[loss] = None
                    
        # print(eval_batch)
        for loss in loss_type :
            if loss == 'ND' :
                summary_metric[loss] =  eval_batch[loss][0]/ eval_batch[loss][1]
            elif loss == 'RMSE' :
                summary_metric[loss] = ((eval_batch[loss][0] / eval_batch[loss][2])**(1/2)) / (eval_batch[loss][1] / eval_batch[loss][2])
            elif loss == 'NLL' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'CRPS' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q10' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q50' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q90' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'ECRPS' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            else :
                summary_metric[loss] = None
                
    return summary_metric

def modelTrain(model: nn.Module,
          optimizer: optim,
          train_loader: DataLoader,
          params: dict,
          epoch: int) -> float :
    
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    _G = Gradient()
    for i, (train_batch, idx, v, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(params["device"])  
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(params["device"])
        v_batch = v.to(torch.float32).to(params["device"])
        idx = idx.permute(1, 0, 2).to(params["device"])

        T, batch_size = labels_batch.shape

        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)
        h, decoder_hidden, decoder_cell = model(train_batch, idx, hidden, cell)

        h = h[-params['predict_steps']:]
        labels_batch = labels_batch[-params['predict_steps']:]

        _step = 0.001 
        alpha_array = torch.arange(start=0, end = 1, step=_step, device = params["device"], requires_grad=True).view(-1, 1, 1).repeat(1, params['predict_steps'], batch_size)
        h = torch.unsqueeze(h, dim = 0).repeat(alpha_array.shape[0], 1, 1, 1)
        y = model.F_inv(h, alpha_array)
        pinball_loss = 2 * (alpha_array - (y > labels_batch).to(torch.float)) * (labels_batch - y) 
        pinball_loss = (pinball_loss[1:] + pinball_loss[:-1]) / 2 * _step

        loss = torch.sum(pinball_loss) / batch_size / params['predict_steps']

        if i % 1000 == 0 :
            print(loss)
    
        loss.backward()
        optimizer.step()
        
        
    return loss_epoch

def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       validation_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim, 
                       params: dict) :

    logger.info('begin training and evaluation')
    best_test_CRPS = float('inf')
    train_len = len(train_loader)
    crps_summary = np.zeros(params["num_epochs"])
    
    evaluation_summary = {}
    loss_summary = np.zeros((train_len * params["num_epochs"]))
    
    for epoch in tqdm(range(params["num_epochs"])):
        logger.info('Epoch {}/{}'.format(epoch + 1, params["num_epochs"]))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = modelTrain(model, optimizer, train_loader, params, epoch)
        
        test_metrics = evaluate(model, validation_loader, params, epoch, loss_type = ['ND', 'RMSE', 'CRPS', 'Q50', 'Q90', 'ECRPS'])

        crps_summary[epoch] = test_metrics['CRPS']
        evaluation_summary[epoch] = test_metrics
        is_best = crps_summary[epoch] <= best_test_CRPS
        
        
        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()},
                          epoch = epoch,
                          is_best = is_best,
                          checkpoint = params["model_dir"],
                          ins_name = f'{params["lstm_hidden_dim"], params["embedding_dim"], params["layers"], params["learning_rate"]}')

        
        
        if is_best:
            logger.info('- Found new best CRPS')
            best_test_CRPS = crps_summary[epoch]
            best_json_path = os.path.join(params["model_dir"], f'ins_{params["lstm_hidden_dim"], params["embedding_dim"], params["layers"], params["learning_rate"]}_metrics_validation_best_weights.json')
            save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best CRPS is: %.5f' % best_test_CRPS)
        last_json_path = os.path.join(params['model_dir'],  f'ins_{params["lstm_hidden_dim"], params["embedding_dim"], params["layers"], params["learning_rate"]}_validation_metric.json')
        
        save_dict_to_json(evaluation_summary, last_json_path)

    return best_test_CRPS



