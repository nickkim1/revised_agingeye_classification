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


# sacrifices spatial relationships, but probably a decent baseline to use  
class FeedForwardNN(nn.Module):
    def __init__(self): 
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_features=2400, out_features=1000, bias=True)
        self.layer2 = nn.Linear(in_features=1000, out_features=10, bias=True)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
    def forward(self, x): 
        x = x.reshape((1,2400))
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.sigmoid(torch.tensor(torch.flatten(x), dtype=torch.float32, requires_grad=True))
        return x

# testing cnn, based on very simple architecture (like deepchrome)
# conv -> maxpool -> dropout -> linear -> linear -> linear layers
# note i'll treat each nucleotide as a feature for the 1d CNN 

class SimpleCNN(nn.Module):
    def __init__(self, simple_config): 
        super(SimpleCNN, self).__init__()
        model_architecture = simple_config['model_architecture']
        self.conv_one = nn.Conv1d(in_channels=simple_config['conv1']['Cin'], out_channels=simple_config['conv1']['Cout'], kernel_size=simple_config['conv1']['kernel_size']) # output dim: (1, 300, 582)
        self.max_pool = nn.MaxPool1d(kernel_size=simple_config['mp1']['kernel_size']) 
        self.dropout = nn.Dropout1d(simple_config['dropout']['probability'])
        self.linear_1 = nn.Linear(in_features=self.calculate_flattened_size(simple_config, model_architecture), out_features=simple_config['fc1']['out'])
        self.linear_2 = nn.Linear(in_features=simple_config['fc1']['out'], out_features=simple_config['fc2']['out'])
        self.linear_3 = nn.Linear(in_features=simple_config['fc2']['out'], out_features=simple_config['fc3']['out'])
    def forward(self, x): 
        x = self.conv_one(x)
        # print('first conv layer: ', x.shape)
        x = F.relu(x)
        x = self.max_pool(x)
        # print('first pooling layer: ', x.shape)
        x = self.dropout(x)
        # print('dropout: ', x.shape)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        # print('post flattening: ', x.shape)
        x = self.linear_1(x)
        # print('first linear: ', x.shape)
        x = F.relu(x)
        x = self.linear_2(x)
        # print('second linear: ', x.shape)
        x = F.relu(x)
        x = self.linear_3(x)
        # print('third linear: ', x.shape)
        x = torch.sigmoid(x)
        # print('sigmoid', x.shape)
        x = torch.squeeze(x) # flatten in order to match the shape of the labels
        # print('final', x.shape)
        return x
    
    def calculate_flattened_size(self, simple_config:dict, model_architecture): 
        list_of_layers = [a.strip() for a in model_architecture.split(",")]
        L_in = 0
        # this only works since the essential formula for mp and also conv are like the same 
        layer_inout_size_queue = collections.deque([])
        for layer_number, layer in enumerate(list_of_layers): 
            if simple_config[layer]: 
                # if you hit this condition you know that you're at the first convolutional layer 
                if layer_number == 0 and 'Lin' in simple_config[layer]: 
                    L_in = simple_config[layer]['Lin']
                else:
                    L_in = layer_inout_size_queue.popleft()
                # you should've hit a convolutional layer before this so pop left
                L_out_new = self.calculate_cnn_mp_output_size(L_in, simple_config, layer)
                # print(L_out_new, simple_config[layer])
                layer_inout_size_queue.append(L_out_new)
        # for the purposes of this prj ig this -2 assumption is safe. Cout should be prev convolutional layer's output size 
        final_input = simple_config[list_of_layers[len(list_of_layers)-2]]['Cout']
        final_output = layer_inout_size_queue.popleft()
        return final_input * final_output
    
    def calculate_cnn_mp_output_size(self, L_in, simple_config, layer):
        return int((L_in + ((2 * simple_config[layer]['padding']) + (-1 * simple_config[layer]['dilation'] * (simple_config[layer]['kernel_size'] - 1))) - 1) / simple_config[layer]['stride'] + 1)