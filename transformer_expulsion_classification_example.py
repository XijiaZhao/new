import torch
import torch.nn as nn
import torch.optim as optim
from transformer_train import Transformer_Train
from data_preparation import TimeSeriesDataPreparation
from hdf5_data_loader import HDF5DataLoader
torch.manual_seed(0)

# Data Loading and Preparation
loader = HDF5DataLoader(r'Dataset/PHS Database [No Padding].h5')
ts_data, headers, metadata = loader.load_data()


# prep = TimeSeriesDataPreparation(ts_data, metadata, input_columns=[0, 1, 2, 3], label_key='Expulsion')
# input_data, labels = prep.get_data_labels()
# train_input, test_input, train_labels, test_labels = prep.split_data(input_data, labels, test_size=0.5)
# print(train_input.shape,test_input.shape,train_labels.shape,test_labels.shape)

# train_loader = prep.get_batches(train_input, train_labels, batch_size=32)
# test_loader = prep.get_batches(test_input, test_labels, batch_size=32)

prep2 = TimeSeriesDataPreparation(ts_data, metadata, input_columns=[0, 1, 2, 3], label_keys=["Stackup#","Expulsion"])
train_input2, train_labels2, selected_unique_labels, test_input2, test_labels2, remaining_unique_labels = prep2.get_data_labels_partial(15, selection_key="Stackup#", output_key="Expulsion")
train_loader2 = prep2.get_batches(train_input2, train_labels2, batch_size=32)
test_loader2 = prep2.get_batches(test_input2, test_labels2, batch_size=32)



transformer =  Transformer_Train(num_features=4, num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.model.parameters(), lr=0.0001)
transformer.train(train_loader2, test_loader2, criterion, optimizer, epochs=100)


