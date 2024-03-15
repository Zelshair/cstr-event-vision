"""
Calculate Dataset Statistics Script.

This script is designed to compute and display key statistics of a specified event-based vision dataset. It calculates the average number of events per sample and the average duration of samples within the dataset. The script supports various dataset formats and event representations by dynamically adapting to the specified dataset's structure.

The statistics are computed across both training and test subsets of the dataset to provide comprehensive insights into the overall dataset characteristics. These insights are valuable for understanding dataset complexity, optimizing data loading, and preprocessing steps.

Usage:
    Run this script directly from the command line. Modify the `dataset_name`, `event_rep`, `channels`, and other settings as required for different datasets or configurations.

Example:
    ```
    python calculate_dataset_statistics.py
    ```

Note:
    - Ensure the dataset is supported and correctly specified in the script.
    - The script assumes access to the dataset's root directory and CSV files listing the samples.
"""


import os
import time
import numpy as np
import pandas as pd
from scripts.data_processing import *
import scipy.io as sio
from utils import *

# simple usage example of the function get_norm_params of a supported dataset
def main():
    # set dataset name
    dataset_name = 'DVS-Gesture'

    # get train dataset
    train_dataset_csv, dataset_root = get_train_dataset_csv(dataset_name)

    # get test dataset
    test_dataset_csv, dataset_root = get_test_dataset_csv(dataset_name)

    start_time = time.time()

    avg_num_events_sample, avg_t_sample = get_dataset_statistics(train_dataset_csv, test_dataset_csv, dataset_name, dataset_root)

    end_time = time.time()

    print('Total time taken: {:.2f} seconds\n'.format(end_time-start_time))
    print('dataset name:', dataset_name)
    print('dataset length:', len(train_dataset_csv) + len(test_dataset_csv))
    print("Average number of events per sample = {:.4f}".format(avg_num_events_sample))
    print("Average sample duration = {:.4f} seconds".format(avg_t_sample/1e6))


# Helper function to load the training dataset csv
def get_train_dataset_csv(dataset_name):
    
    # Check if dataset is supported then get the required paths
    dataset_root, train_csv_path, _ = get_dataset_paths(dataset_name)
    
    # load training dataset
    train_dataset_csv = pd.read_csv(train_csv_path)
            
    return train_dataset_csv, dataset_root

# Helper function to load the testing dataset csv
def get_test_dataset_csv(dataset_name):
    # Check if dataset is supported then get the required paths
    dataset_root, test_csv_path = get_dataset_paths(dataset_name)    

    # load training dataset
    test_dataset_csv = pd.read_csv(test_csv_path)

    return test_dataset_csv, dataset_root

# function to get a specified dataset's statistics including: average events per sample, average sample duration
def get_dataset_statistics(train_dataset_csv, test_dataset_csv, dataset_name, dataset_root):
    """
    Computes average statistics over the specified dataset's training and testing partitions.

    This function calculates the average number of events per sample and the average duration of samples within both the training and testing subsets of a given dataset. It provides a detailed look into the dataset's characteristics, which can be critical for understanding its complexity and for making informed decisions regarding preprocessing and model training strategies.

    Parameters:
        train_dataset_csv (DataFrame): A pandas DataFrame containing paths and metadata for the training dataset samples.
        test_dataset_csv (DataFrame): A pandas DataFrame containing paths and metadata for the testing dataset samples.
        dataset_name (str): The name of the dataset being analyzed, used to tailor the reading and parsing of event data.
        dataset_root (str): The root directory path where the dataset is stored.

    Returns:
        avg_num_events_sample (float): The average number of events per sample across the dataset.
        avg_t_sample (float): The average duration of samples in the dataset, in microseconds.

    Note:
        - The function supports various event-based vision dataset formats through conditional handling based on the dataset name and file extensions.
        - The actual reading and parsing of event data are delegated to dataset-specific functions that must be compatible with the dataset's file format and structure.
    """

    # Initialize lists
    number_events_sample = []
    sample_duration = []

    # find full dataset size
    dataset_size = len(train_dataset_csv) + len(test_dataset_csv)

    dataset_csvs = [train_dataset_csv, test_dataset_csv]

    for dataset_csv in dataset_csvs:
        dataset_size = len(dataset_csv)
        for i in range(dataset_size):
            # need to add check for the dataset size... find approximate size... ask user to verify if enough memory is available.
            # 8 * C => Memory = 8 *m * n * p Bytes. or 8 * H * W * C
            print("Parsing dataset sample: [{}|{}]".format(str(i+1), dataset_size), end='\r')
            
            events_dict = get_events_dict(dataset_csv, dataset_name, dataset_root, i)

            number_events_sample.append(len(events_dict['ts']))
            sample_duration.append(events_dict['ts'][-1])

        print()

    avg_num_events_sample = np.mean(number_events_sample)
    avg_t_sample = np.mean(sample_duration)

    return avg_num_events_sample, avg_t_sample

def get_events_dict(dataset_csv, dataset_name, root_dir, index):
    # read events file path
    events_path = dataset_csv['events_file_path'][index]

    # create path to file
    events_file_path = os.path.join(root_dir, events_path)

    # read events
    events_dict = read_sample(events_file_path, dataset_name)

    return events_dict

def read_sample(events_file, dataset_name):
    file_type = events_file.split('.')[-1]

    if file_type == 'aedat':
        if dataset_name == 'CIFAR10-DVS':
            return dat2mat(events_file)
        
        if dataset_name == 'DVS-Gesture':
            return read_aedat_file(events_file) 
            
    
    elif file_type == 'bin':
        return read_ndataset(events_file)

    elif file_type == 'dat':
        return load_atis_data(events_file)
    
    elif file_type == 'mat':
        events_dict = sio.loadmat(events_file)
        # rename 'pol' key it compatible with this framework
        events_dict['p'] = events_dict.pop('pol')

        # flip events vertically and horizontally
        # (verified by visualizing and comparing with the original paper's visualizations)
        events_dict['y'] = 179 - events_dict['y']
        events_dict['x'] = 239 - events_dict['x']

        return events_dict

if __name__ == "__main__":
    main()