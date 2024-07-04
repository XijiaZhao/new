import h5py
import pandas as pd

class HDF5DataLoader:
    def __init__(self, file_path):
        """
        Initializes the HDF5DataLoader with the specified HDF5 file path.

        :param file_path: Path to the HDF5 file.
        """
        self.file_path = file_path

    def load_data(self):
        """
        Loads data from the HDF5 file, including time series data, headers, and metadata.

        :return: A tuple containing time series data, headers, and metadata.
        """
        with h5py.File(self.file_path, 'r') as file:
            # Extract time series data
            ts_data = [file[dataset_name][:] for dataset_name in file.keys()]

            # Get header names from the first dataset
            first_dataset = file[list(file.keys())[0]]
            ts_header_names = first_dataset.attrs.get('headers', [])

            # Extract metadata, limited to the first 20 items
            self.metadata = {}
            dataset_index = 0  # Start the indexing from 1
            for dataset_name in file.keys():
                dataset_metadata = {key: value for key, value in list(file[dataset_name].attrs.items())[:20]}
                self.metadata[dataset_index] = dataset_metadata
                dataset_index += 1  # Increment index for the next dataset

            return ts_data, ts_header_names, self.metadata
    
    def extract_metadata_values(self, keys):
        """
        Extracts values for specified keys from all datasets' metadata.

        :param keys: List of keys to extract values for.
        :return: Dictionary of lists containing values for each key.
        """
        labels = {key: [] for key in keys}
        for index in self.metadata:
            for key in keys:
                labels[key].append(self.metadata[index].get(key, None))
        
        return labels
    
    def export_metadata_to_excel(self, metadata, file_name):
        """Export metadata to an Excel file.

        :param metadata: A list of dictionaries containing metadata. Each dictionary represents a sample.
        :param file_name: Name of the Excel file to save.
        """
        # Convert the metadata from a dictionary of dictionaries to a list of dictionaries
        metadata_list = list(self.metadata.values())
        
        # Convert metadata to DataFrame. Each dictionary in the list becomes one row.
        df = pd.DataFrame(metadata_list)

        # Export to Excel. The DataFrame automatically uses dictionary keys as column headers.
        df.to_excel(f"{file_name}.xlsx", index=False)

# Example usage:
# loader = HDF5DataLoader(r'C:\Users\Edward Wang\Google Drive\Python\RSW SSL\Dataset\PHS Database [No Padding].h5')
# time_series_data, headers, metadata = loader.load_data()
# for index, dataset in enumerate(time_series_data):
#     print(f"Shape of dataset {index}: {dataset.shape}")
# values = loader.extract_metadata_values(['Expulsion', 'AnotherKey'])
# print(values.get('Expulsion'))
