#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

from train import *


# In[2]:

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

logger = logging.getLogger('NQF-RNN.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=['M4-hourly'], type=str, nargs='+')
parser.add_argument('--data_folder', default='dataset\\data', help='Parent dir of the dataset')
parser.add_argument('--model_name', default='model', help='Directory containing params.json')
parser.add_argument('--params_dir', default='params', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


                 
if __name__ == '__main__':
    
    args = parser.parse_args()

    if args.dataset in ['elect', 'traffic', 'solar', 'exchange', 'parts', 'M4-hourly', 'tourism-monthly', 'tourism-quarterly'] :
        dataset_list = [args.dataset]

    else :
        dataset_list = args.dataset 


    for dataset in dataset_list :

        params_path = os.path.join(args.params_dir, dataset, 'params.json')
        data_dir = os.path.join(args.data_folder, dataset)
        model_dir = os.path.join(args.model_name, dataset)
        
        with open(params_path) as f:
            params = json.load(f)

        
        params["relative_metrics"] = args.relative_metrics
        params["sampling"] =  args.sampling
        params["model_dir"] = model_dir
        params["dataset"] = dataset
        params['random_seed'] = 0
            
        logger.info(f"Random seed : {params['random_seed']}")


        logger.info('Loading the datasets...')

        train_set = TrainValDataset(data_dir, dataset, params['num_class'], random_seed = params['random_seed'], datatype = 'train')
        validation_set = TrainValDataset(data_dir, dataset, params['num_class'], random_seed = params['random_seed'], datatype = 'validation')
        test_set = TestDataset(data_dir, dataset, params['num_class'])

        sampler = WeightedSampler(data_dir, dataset, random_seed = params['random_seed']) # Use weighted sampler instead of random sampler

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], sampler=RandomSampler(train_set), num_workers=4)
        validation_loader = DataLoader(validation_set, batch_size=params['predict_batch'], sampler=RandomSampler(validation_set), num_workers=4)
        test_loader = DataLoader(test_set, batch_size=params['predict_batch'], sampler=RandomSampler(test_set), num_workers=4)

        logger.info('Loading complete.')
        best_test_CRPS = float('inf')
        best_params = None
        
        cuda_exist = torch.cuda.is_available()
        _dir = os.path.join(args.model_name, 'best_params.json')

        if dataset in ['exchange', 'parts'] :
            embedding_dim_list = [1, 5, 20]
        else :
            embedding_dim_list = [5, 20]
            
        lstm_hidden_dim_list = [20, 40, 60]
        learning_rate_list = [0.001, 0.005]
        layers_list = [[64,32,16,8,4], [16, 8, 4]]

        logger.info(f'Hyperparameter search : lstm_hidden_dim {lstm_hidden_dim_list}, embedding_dim {embedding_dim_list}, learning_rate {learning_rate_list}, layers {layers_list}')

        for lstm_hidden_dim in lstm_hidden_dim_list :
            for embedding_dim in embedding_dim_list :
                for learning_rate in learning_rate_list :
                    for layers in layers_list :

                            params['layers'] = layers
                            params["learning_rate"] = learning_rate
                            params["lstm_hidden_dim"] = lstm_hidden_dim
                            params["embedding_dim"] = embedding_dim

                            cuda_exist = torch.cuda.is_available()
                                
                            while True :

                                if cuda_exist:
                                    params["device"] = torch.device('cuda')
                                    logger.info('Using Cuda...')
                                    model = Net(params).cuda()

                                else:
                                    params["device"]  = torch.device('cpu')
                                    logger.info('Not using cuda...')
                                    model = Net(params)

                                print(model.model_init(train_loader))
                                if model.model_init(train_loader) <= 1e3 :
                                    break

                            logger.info(params)

                            logger.info(f'Model: \n{str(model)}')

                            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

                            logger.info('Starting training for {} epoch(s)'.format(params["num_epochs"]))

                            test_CRPS = train_and_evaluate(model,
                                            train_loader,
                                            validation_loader,
                                            test_loader,
                                            optimizer,
                                            params)
                            
                            if test_CRPS <= best_test_CRPS :
                                print(test_CRPS, 'best CRPS')
                                print(best_test_CRPS, 'previous best CRPS')
                                best_test_CRPS = test_CRPS
                                best_params = deepcopy(params)

                                best_params["device"] = 0
                                best_params["best CRPS"] = best_test_CRPS
                                print(best_params, 'best params')
                                json_path = os.path.join(params['model_dir'],  'best_params.json')
                                save_dict_to_json(best_params, json_path)