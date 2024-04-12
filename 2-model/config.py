import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import collections
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch.utils.data as torch_data
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd 
import numpy as np
import time as time
import argparse as ap
import data_aug as aug
from models import SimpleCNN, FeedForwardNN, BassetCNN

# TODO: convert this to some sort of JSON format or something idk 

basset_cnn_config = {
}

ff_nn_config = {
}

simple_config = {
    'model_architecture': 'conv1, mp1', 
    'conv1': {
        'Cin': 4,
        'Lin': 100, 
        'Cout': 500, 
        'kernel_size': 20, 
        'stride': 1, 
        'dilation': 1, 
        'padding': 0
    }, 
    'mp1': {
        'kernel_size': 19, 
        'stride': 19, 
        'dilation': 1, 
        'padding': 0
    },
    'dropout': {
        'probability': 0.4
    }, 
    'fc1': {
        'out': 1500
    }, 
    'fc2': {
        'out': 300
    }, 
    'fc3': {
        'out': 1
    }
}

model_config = {
    'DeepChrome': SimpleCNN(simple_config),  
    'MLP': FeedForwardNN(),
    'Basset': BassetCNN()
}

def populate_settings(train_filepath, test_filepath, seqlen, loss, optim, model_type, batch_size): 
    data_config = {
        'train': train_filepath, 
        'test': test_filepath, 
        'seqlen': seqlen
    }
    hyperparameter_config = {
        'learning_rate': 0.001, 
        'num_epochs': 50, 
        'batch_size': batch_size
    }
    optim_config = {
        'Adam': torch.optim.Adam(model_config[model_type].parameters(), lr=hyperparameter_config['learning_rate']), 
        'RMSprop': torch.optim.RMSprop(model_config[model_type].parameters(), lr=hyperparameter_config['learning_rate']), 
        'SGD': torch.optim.SGD(model_config[model_type].parameters(), lr=hyperparameter_config['learning_rate']) 
    }
    loss_config = {
        'BCE': nn.BCELoss() 
    }
    settings = {
        'data': data_config,
        'optim_name': optim,
        'model_name': model_type,
        'model': model_config[model_type], 
        'hyperparameters': hyperparameter_config, 
        'optimizer': optim_config[optim], 
        'loss': loss_config[loss], 
    }
    return settings

