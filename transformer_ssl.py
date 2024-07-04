import torch
import torch.nn as nn
import torch.optim as optim
from transformer_train import SSL_Transformer_Train, Classification_Train
from data_preparation import TimeSeriesDataPreparation, CustomDataset
from hdf5_data_loader import HDF5DataLoader
from data_preparation import StackupPairsDataset
from ssl_evaluation import ModelEvaluator
torch.manual_seed(0)

# Data Loading and Preparation
loader = HDF5DataLoader(r'Dataset/PHS Database [No Padding].h5')
ts_data, headers, metadata = loader.load_data()

#################################### SSL Training 
prep = TimeSeriesDataPreparation(ts_data, metadata, input_columns=[0, 1, 2, 3], label_keys='Stackup#')
input_data, labels = prep.get_data_labels() 
# input data: (1142, 260, 4)
# labels: the stack up information, (1142,)

dataset = StackupPairsDataset(input_data, labels)
train_pairs, test_pairs, train_labels, test_labels = dataset.split_data(test_size=0.2)
# the train pairs are just pairs of index, labels are the according pairs are positive or negative
train_loader, test_loader = dataset.create_dataloaders(train_pairs, test_pairs, train_labels, test_labels, batch_size =32)
# the train_loader, see at the get_items() of the subset, it gets ((data[pair0],data[pair1]),label)
SSL_trainer =  SSL_Transformer_Train(num_features=4)
SSL_trainer.train(train_loader, test_loader, lr=0.0001, num_epochs=1000)
trained_ssl_model = SSL_trainer.get_model()

torch.save(trained_ssl_model, 'ssl_model_1000_new.pth')

#################################data preparation for training the fully connected layers
prep2 = TimeSeriesDataPreparation(ts_data, metadata, input_columns=[0, 1, 2, 3], label_keys="Expulsion")
input_data, labels = prep2.get_data_labels()
train_input2, test_input2, train_labels2, test_labels2 = prep2.split_data(input_data, labels, test_size=0.5)
train_loader2 = prep2.get_batches(train_input2, train_labels2, batch_size=32)
test_loader2 = prep2.get_batches(test_input2, test_labels2, batch_size=32)

#################################partial data preparation on selected Stackup# for training the fully connected layers
# prep2 = TimeSeriesDataPreparation(ts_data, metadata, input_columns=[0, 1, 2, 3], label_keys=["Stackup#","Expulsion"])
# train_input2, train_labels2, selected_unique_labels, test_input2, test_labels2, remaining_unique_labels = prep2.get_data_labels_partial(15, selection_key="Stackup#", output_key="Expulsion")
# train_loader2 = prep2.get_batches(train_input2, train_labels2, batch_size=32)
# test_loader2 = prep2.get_batches(test_input2, test_labels2, batch_size=32)
# print(f'selected labels: {selected_unique_labels}')
# print(f'remaining unique labels: {remaining_unique_labels}')

################################SSL t-SNE plot
trained_ssl_model = torch.load('ssl_model_1000.pth')
# label_mapping = CustomDataset(train_input2, train_labels2).get_label_mapping()
# eval = ModelEvaluator(trained_ssl_model, test_loader2,label_mapping)
# eval.visualize_with_tsne('Stackup clustering, after 1000 SSL training epochs')

################################ bulid and train the classifier upon the pre-trained SSL transformer
transformer =  Classification_Train(trained_ssl_model, num_classes=4, feature_size=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.model.parameters(), lr=0.0001)
transformer.train(train_loader2, test_loader2, criterion, optimizer, epochs=100)
# transformer.print_model_trainable_parameters()
torch.save(transformer, 'classfication_model_1000.pth')

############################### classification confusion matrix
transformer = torch.load('classfication_model_1000.pth')
transformer.plot_confusion_matrix(test_loader2, num_classes=4)


