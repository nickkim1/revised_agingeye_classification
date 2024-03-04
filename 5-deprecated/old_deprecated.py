

# ====================================== OLD CODE  ======================================
# train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=BSampler(num_rows=len(train_dataset), gene_ids=train_dataset.gene_ids_and_indices()[0], indices=train_dataset.gene_ids_and_indices()[1], batch_size=1))
# test_dataloader = DataLoader(dataset=test_dataset, batch_sampler=BSampler(num_rows=len(test_dataset), gene_ids=test_dataset.gene_ids_and_indices()[0], indices=test_dataset.gene_ids_and_indices()[1], batch_size=1))

#  custom batch sampler class (for generating the 4 x 600 inputs) et
# class BSampler(BatchSampler): 
#     def __init__(self, num_rows, gene_ids, indices, batch_size):
#         super(BSampler, self).__init__(train_dataset, batch_size, drop_last=False) # i forget why dataset is needed here 
#         self.gene_ids = gene_ids
#         self.num_rows = num_rows
#         self.indices = indices 
#     def __iter__(self):
#         np.random.seed(0)
#         # np.random.shuffle(self.gene_ids)
#         batches = []
#         for ignore in range(self.num_rows):
#             # randomly choose an idx (sample) from 0 to the full dataset 
#             batch = [random.choice(self.gene_ids)]
#             batches.append(batch)
#         return iter(batches)
#     def __len__(self):
#         # this doesn't return anything informative unless i change the num_rows into constructor param
#         return self.num_rows

# define the CNN architecture for basset
# class BassetCNN(nn.Module):
#     def __init__(self): 
#         super(BassetCNN, self).__init__()
#         self.conv_one = nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19)
#         self.batchnorm_one = nn.BatchNorm1d(300) # input = num channels 
#         self.pool_one = nn.MaxPool1d(kernel_size=3) 
#         self.conv_two = nn.Conv1d(in_channels=300, out_channels=200, kernel_size=11)
#         self.batchnorm_two = nn.BatchNorm1d(200) # input = num channels 
#         self.pool_two = nn.MaxPool1d(4)
#         self.conv_three = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7)
#         self.batchnorm_three = nn.BatchNorm1d(200) # input = num channels 
#         self.pool_three = nn.MaxPool1d(kernel_size=4)
#         self.fc1 = nn.Linear(2000, 1000) # 1000 unit linear layer (same as paper)
#         self.dropout_one = nn.Dropout1d(0.3)  
#         self.fc2 = nn.Linear(1000, 1000) # 
#         self.dropout_two = nn.Dropout1d(0.3)
#         self.fc3 = nn.Linear(1000, 1) # output dim should be [1x164] since i unsqueezed @ flattening above
#     def forward(self, x): 
#         x = self.conv_one(x)
#         x = self.batchnorm_one(x)
#         x = F.relu(x)
#         x = self.pool_one(x)
#         x = self.conv_two(x)
#         x = self.batchnorm_two(x)
#         x = F.relu(x)
#         x = self.pool_two(x)
#         x = self.conv_three(x)
#         x = self.batchnorm_three(x)
#         x = F.relu(x)
#         x = self.pool_three(x)
#         x = self.fc1(x.unsqueeze(0)) # unsqueeze after flattening
#         x = F.relu(x)
#         x = self.dropout_one(x)
#         x = self.fc2(x)
#         x = self.dropout_two(x)
#         x = self.fc3(x)
#         x = F.sigmoid(torch.tensor(torch.flatten(x), dtype=torch.float32, requires_grad=True))
#         return x


# if (predicted.item() > 0.5 and labels.item() == 1) or (predicted.item() <= 0.5 and labels.item() == 0): 
#     t_accuracy_per_batch.append(1)
# elif (predicted.item() > 0.5 and labels.item() == 0) or (predicted.item() <= 0.5 and labels.item() == 1): 
#     t_accuracy_per_batch.append(0)
# every other epoch, reverse complement the sample 
# if epoch % 2 == 1: 
#     # samples = data_aug.reverse_complement_sequence(data_aug, samples)
#     samples = data_aug.reverse_complement_sequence(data_aug, samples)
    
# if (predicted.item() > 0.5 and labels.item() == 1) or (predicted.item() <= 0.5 and labels.item() == 0): 
#     t_accuracy_per_batch.append(1)
# elif (predicted.item() > 0.5 and labels.item() == 0) or (predicted.item() <= 0.5 and labels.item() == 1): 
#     t_accuracy_per_batch.append(0)
    
# int((((simple_config['num_samples_per_batch']-simple_config['conv_1_kernel_size'])*simple_config['batch_size'])/simple_config['pool_kernel_size'])*simple_config['conv_1_out']

# x = torch.zeros(size=[256, 4, 100])
# conv1 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=50)
# x = conv1(x)
# print(x.shape)
# mp1 = nn.MaxPool1d(10)
# x= mp1(x)
# print(x.shape)
    
# predicted = torch.where(predicted > 0.5, 1.0, 0.0)
    
# class CustomDataset2(Dataset): 
# def __init__(self, dataset, n_samples): 
#     self.n_samples = n_samples
#     self.features = []
#     self.labels = []
#     # iterate over all the rows of the training dataset (training examples)
#     for i in range(self.n_samples): 
#         s = np.zeros([sequence_length, 4]) 
#         sequence = dataset[0][i]
#         label = dataset[1][i]
#         # iterate over each base within the sequence for that training example 
#         for j in range(sequence_length): 
#             base_at_jth_position = self.one_hot_encode(0, sequence[j]) # one hot encode sequences as: A T C G (top -> bottom)
#             s[j][base_at_jth_position] = 1
#         self.features.append(s)
#         self.labels.append(label)
# def __len__(self):
#     return self.n_samples
# def __getitem__(self, index):
#     return self.features[index], self.labels[index]
# def one_hot_encode(self, marker, base):
#     if base.upper() == 'A': marker = 0
#     elif base.upper() == 'T': marker = 1
#     elif base.upper() == 'C': marker = 2
#     elif base.upper() == 'G': marker = 3
#     return marker

# class BSampler(BatchSampler): 
#     def __init__(self, num_rows_in_train, gene_ids, indices, batch_size):
#         super(BSampler, self).__init__(train_dataset, batch_size, drop_last=False) # i forget why dataset is needed here 
#         self.gene_ids = gene_ids
#         self.indices = indices 
#         self.num_batches = int(num_rows_in_train / batch_size)
#         self.batch_size = batch_size
#     def __iter__(self):
#         batches = []
#         for _ in range(self.num_batches):
#             batch = []
#             batch_gene = random.choice(self.gene_ids) 
#             batches.append(batch)
#         return iter(batches)
#     def __len__(self):
#         # this doesn't return anything informative unless i change the num_rows into constructor param
#         return self.num_rows

# for running with the debugger 
# model_type = 'testing'
# train_filepath = '../0-data/2-final/1-finalcsv/agingeye/cts_bed_train.csv'
# train_filepath = '/Users/nicolaskim/Desktop/research/singh_lab/basset_basenji/aging_eye_classification/0-data/2-final/1-finalcsv/agingeye/cts_bed_train.csv'
# test_filepath = '../0-data/2-final/1-finalcsv/agingeye/cts_bed_test.csv'
# test_filepath = '/Users/nicolaskim/Desktop/research/singh_lab/basset_basenji/aging_eye_classification/0-data/2-final/1-finalcsv/agingeye/cts_bed_test.csv'

# TODO 1: (x meeting)
# * this is not the full implementation of the Basset architecture
# * no need to preprocess because this is just bulk data, not single cell (can just use raw counts)
# * idk what more to do besides just using a different architecture 
# - control data vs. exp data - two separate models 
# - make each figure take in a description field automatically filled in w/ model's relevant params at time of running <-- 

# TODO 2: (x meeting)
# augment the data by reverse complementing the data, inverting it, shifting it 

# TODO 3: set up a config dict for downstream use 

# TODO 4 (Jan 29 meeting): 
# 1. First check preprocessing 
    # - check if getFasta and the subsequent up/down regions on the chromosome are fetching the right things 
    # - check if one hot encoding is correct
    # - check if i'm fetching the correct TSS sites (use the experimentally validated TSS sites for testing)
# 2. Next check if the model is convolving correctly 
    # - try with smaller kernel size and smaller input size (ex: 4 x 10 input with kernel size of 5, and manually calculate to see if output shape is correct)
# 3. Try different model architecture, i.e. maybe just one dropout and one pooling layer 
# 4. If the RNAseq data fails, then try to make the ATAC-seq training work 
    # - implement above suggestions for the ATAC-seq data trained model, but in terms of input data, input the sequences corresponding to whole genes (not binned intervals)
# > If all else fails, try the CAGE dataset instead of RNAseq data 
# 5. IFF 1-4/> have been tried and done then scale the output of the model so that it oeprates for >1 dataset input 

# TODO: 
# plot the distribution of the counts 
# also have functionality (a wget script) for getting all necessary files loaded in for analysis

# BUGS
# 1. I got this weird ass float error so I just am working w toy data + shorter sequences
    # Might also help with debugging internals of the model

# print dimensions for reference
# print('===============================================================================')
# print("This is the shape of the training data: ", train_data.shape)
# print("This is a brief sample of the training data: ", train_data[:2])
# print("This is the shape of the test data: ", test_data.shape)
# print('===============================================================================')