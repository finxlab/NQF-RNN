#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from train import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import easydict


# In[5]:


logger = logging.getLogger('NQF-RNN.Test')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=['elect', 'traffic', 'solar', 'M4-hourly', 'tourism-Monthly', 'tourism-Quarterly'], type=str, nargs='+')
parser.add_argument('--data_folder', default='dataset/data', help='Parent dir of the dataset')
parser.add_argument('--model_name', default='model', help='Directory containing params.json')
parser.add_argument('--params_dir', default='params', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--ins_num', default = '0')


def evaluate_test(model, test_loader, params, loss_type = ['ND']):
    '''
    Args:
        model: (torch.nn.Module) the neural network
        test_loader: load test data and labels
        params: (Params) hyperparameters
    Returns:
        summary_metric : (dictionary) evaluation
    '''
    
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
            elif loss == 'Q1' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q5' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q95' :
                eval_batch[loss] = np.zeros(2)
            elif loss == 'Q99' :
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
            
            
            samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling_n = params['sample_times'])
            if i== 0 :

                print(sample_mu)
                print(labels_batch[params['predict_start']:])
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

                elif loss == 'Q1' :
                    upper, lower = accuracy_QLOSS(samples, labels_batch[params['predict_start']:], q = 0.01)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'Q5' :
                    upper, lower = accuracy_QLOSS(samples, labels_batch[params['predict_start']:], q = 0.05)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'Q95' :
                    upper, lower = accuracy_QLOSS(samples, labels_batch[params['predict_start']:], q = 0.95)
                    eval_batch[loss][0] += upper
                    eval_batch[loss][1] += lower

                elif loss == 'Q99' :
                    upper, lower = accuracy_QLOSS(samples, labels_batch[params['predict_start']:], q = 0.99)
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
            elif loss == 'Q1' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q5' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
            elif loss == 'Q95' :
                summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1]
            elif loss == 'Q99' :
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
    


# In[37]:


if __name__ == '__main__':

    args = parser.parse_args()
    

    if args.dataset in ['elect', 'traffic', 'solar', 'M4-hourly', 'tourism-monthly', 'tourism-quarterly'] :
        dataset_list = [args.dataset]

    else :
        dataset_list = args.dataset 


    for n in range(10):
        for dataset in dataset_list :
            data_dir = os.path.join(args.data_folder, dataset)
            model_dir = os.path.join(args.model_name, dataset)
            params_path = os.path.join(model_dir, 'best_params.json')

            with open(params_path) as f:
                params = json.load(f)
            params["model_dir"] = model_dir

            cuda_exist = torch.cuda.is_available()

            # Set random seeds for reproducible experiments if necessary
            if cuda_exist:
                params["device"] = torch.device('cuda')
                logger.info('Using Cuda...')
                model = Net(params).cuda()
                
            else:
                params["device"]  = torch.device('cpu')
                logger.info('Not using cuda...')
                model = Net(params)

            logger.info(params)

            logger.info('Loading the test datasets...')
            
            test_set = TestDataset(data_dir, dataset, params['num_class'])
            test_loader = DataLoader(test_set, batch_size=params['predict_batch'], sampler=None, num_workers=4)
            logger.info('Loading complete.')
                    
            best_json_path = os.path.join(params["model_dir"], f'ins_{params["lstm_hidden_dim"], params["embedding_dim"], params["layers"], params["learning_rate"]}_best.pth.tar')
            load_checkpoint(best_json_path, model)
                
            evaluation_summary = evaluate_test(model, test_loader, params, loss_type = ['ND', 'RMSE',  'CRPS', 'Q1', 'Q5', 'Q10', 'Q50', 'Q90', 'Q95', 'Q99', 'ECRPS'])
            last_json_path = os.path.join(params['model_dir'],  f'best_test_metric_{n}.json')
            save_dict_to_json(evaluation_summary, last_json_path)
    

