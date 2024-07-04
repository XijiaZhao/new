import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import torch
np.random.seed(0)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        # Encode labels from strings to integers
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.encoded_labels = [self.label_mapping[label] for label in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return data_tensor, label_tensor
    
    def get_label_mapping(self):
        """
        Returns the label mapping (from string to integer).
        """
        return self.label_mapping

    def reverse_label_mapping(self):
        """
        Returns a reverse mapping from integer to original string labels.
        """
        return {v: k for k, v in self.label_mapping.items()}

    def get_original_labels(self):
        """
        Returns the original string labels.
        """
        return self.original_labels

class TimeSeriesDataPreparation:
    def __init__(self, ts_data, metadata, input_columns, label_keys, max_length=None):
        self.ts_data = ts_data  # Tuple of arrays
        self.metadata = metadata
        self.input_columns = input_columns
        self.label_keys = label_keys
        self.max_length = max_length or self.determine_max_length()

    def determine_max_length(self):
        """Determines the maximum sequence length in the dataset."""
        return max(sequence.shape[0] for sequence in self.ts_data)

    def pad_sequences(self, sequences, max_length):
        """Pads sequences to the specified max_length."""
        padded_sequences = np.zeros((len(sequences), max_length, len(self.input_columns)))
        for i, sequence in enumerate(sequences):
            length = min(len(sequence), max_length)
            padded_sequences[i, :length, :] = sequence[:length, self.input_columns]
        return padded_sequences
    
    def min_max_normalize(self, data):
        """Applies Min-Max normalization to the data."""
        # Initialize an array to store the normalized data
        normalized_data = np.zeros_like(data)

        # Iterate over each channel (feature) to normalize
        for channel in range(data.shape[2]):
            min_val = np.min(data[:, :, channel])
            max_val = np.max(data[:, :, channel])

            # Perform Min-Max normalization for each channel
            normalized_data[:, :, channel] = (data[:, :, channel] - min_val) / (max_val - min_val)

        return normalized_data

    def get_data_labels(self):
        """Extracts and pads input data, and extracts labels."""
        # Pad input data to max_length
        input_data = self.pad_sequences(self.ts_data, self.max_length)
        input_data = self.min_max_normalize(input_data)

        # Extract labels for multiple keys
        labels = []
        for idx in range(len(self.ts_data)):
            label_values = self.metadata[idx].get(self.label_keys) 
            labels.append(label_values)

        # Convert labels list to a NumPy array
        labels_array = np.array(labels)
        return input_data, labels_array
    
    def get_data_labels_partial(self, num_unique_labels, selection_key, output_key):
        """
        Extracts and pads input data, and extracts labels for a random subset of unique label categories based on selection_key.
        Returns labels corresponding to output_key for the selected and remaining data.
        """
        if num_unique_labels <= 0:
            raise ValueError("num_unique_labels must be positive.")

        if selection_key not in self.label_keys or output_key not in self.label_keys:
            raise ValueError("Both selection_key and output_key must be in the provided label_keys.")

        # Extract initial data and selection_key labels
        input_data = self.pad_sequences(self.ts_data, self.max_length)
        input_data = self.min_max_normalize(input_data)

        selection_labels = np.array([self.metadata[idx].get(selection_key) for idx in range(len(self.ts_data))])
        output_labels = np.array([self.metadata[idx].get(output_key) for idx in range(len(self.ts_data))])

        # Find unique labels based on selection_key
        unique_labels = np.unique(selection_labels)
        if num_unique_labels > len(unique_labels):
            raise ValueError(f"Requested number of unique labels ({num_unique_labels}) exceeds the total number of unique labels available ({len(unique_labels)}).")

        # Randomly select a subset of the unique labels
        selected_labels = np.random.choice(unique_labels, size=num_unique_labels, replace=False)

        # Filter data and labels
        selected_indices = np.isin(selection_labels, selected_labels)
        remaining_indices = ~selected_indices

        selected_data = input_data[selected_indices]
        selected_output_labels = output_labels[selected_indices]

        remaining_data = input_data[remaining_indices]
        remaining_output_labels = output_labels[remaining_indices]
        remaining_unique_labels = np.setdiff1d(unique_labels, selected_labels)

        return selected_data, selected_output_labels, selected_labels, remaining_data, remaining_output_labels, remaining_unique_labels

    def split_data(self, input_data, labels, test_size=0.2):
        """
        Splits the data into training and testing sets.

        :param input_data: Input data as a NumPy array.
        :param labels: Labels as a NumPy array.
        :param test_size: Proportion of the dataset to include in the test split.
        :return: Tuple of (train_input, test_input, train_labels, test_labels)
        """
        train_input, test_input, train_labels, test_labels = train_test_split(
            input_data, labels, test_size=test_size, random_state=42
        )
        return train_input, test_input, train_labels, test_labels


    def get_batches(self, data, labels, batch_size):
        """
        Generates batches of data.

        :param data: Input data as a NumPy array.
        :param labels: Labels as a NumPy array.
        :param batch_size: Size of each batch.
        :yield: Batches of (batch_data, batch_labels)
        """
        dataset = CustomDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

# Example usage:
# from hdf5_data_loader import HDF5DataLoader
# from data_preparation import TimeSeriesDataPreparation

# loader = HDF5DataLoader('your_file_path.h5')
# ts_data, headers, metadata = loader.load_data()

# prep = TimeSeriesDataPreparation(ts_data, metadata, input_columns=[0, 1, 2, 3], label_key='YourLabelKey')
# input_data, labels = prep.get_data_labels()
# train_input, test_input, train_labels, test_labels = prep.split_data(input_data, labels)

# batch_size = 32
# for batch_data, batch_labels in prep.get_batches(train_input, train_labels, batch_size):
#     # Train your model on each batch_data, batch_labels

class StackupPairsDataset(Dataset):
    def __init__(self, data, stackup_labels):
        self.data = data
        self.stackup_labels = stackup_labels
        self.pairs, self.labels = self.create_pairs()

    def create_pairs(self):
        label_to_indices = {label: np.where(self.stackup_labels == label)[0] for label in np.unique(self.stackup_labels)}

        positive_pairs = []
        negative_pairs = []

        # transverse all the individual labels in stackup_labels
        for idx, label in enumerate(self.stackup_labels):
            # Positive pair creation # randomly select a sample in the same stackup label 
            positive_index = idx
            # to make sure the randomly selected positive_index is different from the current index 'idx', if they are the same,\
            # the loop will keep going
            while positive_index == idx:
                positive_index = np.random.choice(label_to_indices[label])
            positive_pairs.append([idx, positive_index])

            # Negative pair creation
            # get another label apart from the current positive label, e.g., one of 76/77/78... if current stackup is 75 (unique)
            negative_label = np.random.choice(list(set(self.stackup_labels) - set([label])))
            # select a index fromt the selected label (multiple samples under the same label)
            negative_index = np.random.choice(label_to_indices[negative_label])
            # add the negative index to the negative pairs, it is negative with the current idx
            negative_pairs.append([idx, negative_index])

        positive_pairs = np.array(positive_pairs)
        negative_pairs = np.array(negative_pairs)

        # Combine positive and negative pairs
        
        all_pairs = np.vstack((positive_pairs, negative_pairs)) #(2284,2), first half is the positive pairs
        all_labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs), dtype=np.float32) #the first half is 1, which is the positive pairs

        # Shuffle pairs and labels in the same order
        shuffle_indices = np.random.permutation(len(all_pairs))
        all_pairs = all_pairs[shuffle_indices]
        all_labels = all_labels[shuffle_indices]

        return all_pairs, all_labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        index1, index2 = self.pairs[idx]
        tsdata1, tsdata2 = self.data[index1], self.data[index2]
        label = self.labels[idx]
        return (tsdata1, tsdata2), label
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the pairs and labels into training and testing subsets.

        :param test_size: Fraction of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: Four tuples containing training pairs, testing pairs, training labels, and testing labels.
        """
        train_pairs, test_pairs, train_labels, test_labels = train_test_split(
            self.pairs, self.labels, test_size=test_size, random_state=random_state
        )
        return train_pairs, test_pairs, train_labels, test_labels    


    def create_dataloaders(self, train_pairs, test_pairs, train_labels, test_labels, batch_size, shuffle=True):
        """
        Creates DataLoader instances for training and testing datasets.
        """
        # Create training and testing subsets
        train_dataset = SubsetStackupPairs(self, train_pairs, train_labels)
        test_dataset = SubsetStackupPairs(self, test_pairs, test_labels)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

class SubsetStackupPairs(Dataset):
    """
    Subset of StackupPairsDataset to handle training and testing datasets separately.
    """
    def __init__(self, parent_dataset, pairs, labels):
        self.parent_dataset = parent_dataset
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        index1, index2 = self.pairs[idx]
        data1, data2 = self.parent_dataset.data[index1], self.parent_dataset.data[index2]
        label = self.labels[idx]
        # Convert data to Float format
        index1, index2 = self.pairs[idx]
        data1, data2 = self.parent_dataset.data[index1], self.parent_dataset.data[index2]
        label = self.labels[idx]

        # Convert numpy arrays to PyTorch tensors and to Float format
        data1 = torch.from_numpy(data1).float()
        data2 = torch.from_numpy(data2).float()

        # Convert string labels to numerical format (e.g., 0 or 1) and then to tensor
        label= torch.tensor(label, dtype=torch.float32)
        return (data1, data2), label
# Example usage:
# Instantiate the dataset
# dataset = StackupPairsDataset(data, stackup_labels)
# train_pairs, test_pairs, train_labels, test_labels = dataset.split_data(test_size=0.2)
# train_loader, test_loader = dataset.create_dataloaders(train_pairs, test_pairs, train_labels, test_labels, batch_size =32)



    