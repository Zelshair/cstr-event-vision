"""
Configuration for supported datasets in this project.

This module specifies the supported datasets and organizes their configuration, including
paths to training and testing CSV filenames and the dataset root directories. It facilitates
the dynamic loading of datasets through a unified interface by mapping dataset names to their
corresponding loader functions.

The configuration is structured to support easy addition, removal, and modification of datasets,
promoting scalability and maintainability of the dataset management system.

To add another dataset, simply add a dictionary entry in 'dataset_config' containing the
relative root directory, training file names, and testing file name. Then add the dataset
initialization function to 'dataset_functions'.
"""

# Import dataset loader functions
from scripts.datasets import *

# Unified dataset configuration detailing CSV paths and root directories for each supported dataset.
dataset_config = {
    'N-MNIST': {
        'train_csv': "N-MNIST_train.csv",
        'test_csv': "N-MNIST_test.csv",
        'root': '/data/Datasets/Event-Based/N-MNIST/'
    },
    'N-Cars': {
        'train_csv': 'Prophesee_Dataset_n_cars_n-cars_train.csv',
        'test_csv': 'Prophesee_Dataset_n_cars_n-cars_test.csv',
        'root': '/data/Datasets/Event-Based/N-Cars/Prophesee_Dataset_n_cars'
    },
    'N-Caltech101': {
        'train_csv': 'Caltech101_train_2.csv',
        'test_csv': 'Caltech101_test_2.csv',
        'root': '../Datasets/N-Caltech101/Caltech101/'
    },
    'CIFAR10-DVS': {
        'train_csv': 'CIFAR10-DVS_train_90_10.csv',
        'test_csv': 'CIFAR10-DVS_test_90_10.csv',
        'root': '../Datasets/CIFAR10-DVS/'
    },
    'ASL-DVS': {
        'train_csv': 'ASL-DVS_train.csv',
        'test_csv': 'ASL-DVS_test.csv',
        'root': '../Datasets/ASL-DVS/'
    },
    'DVS-Gesture': {
        'train_csv': 'dvs_gesture_500_250v2_train.csv',
        'test_csv': 'dvs_gesture_500_250v2_test.csv',
        'root': '../Datasets/dvs_gesture_500_250v2/'
    }
    # Add additional datasets configurations as needed.
}

# Mapping of dataset names to their respective loader functions for dynamic dataset initialization.
dataset_functions = {
    'N-MNIST': NMNIST,
    'N-Caltech101': NCaltech101,
    'N-Cars': NCars,
    'CIFAR10-DVS': CIFAR10DVS,
    'ASL-DVS': ASLDVS,
    'DVS-Gesture': DVSGesture,
    # Extend with other dataset loader functions as required.
}

# Ensure that every dataset defined in dataset_config has a corresponding loader function in dataset_functions.
# This assertion helps maintain consistency and prevent runtime errors due to missing configurations or functions.
assert set(dataset_functions.keys()) == set(dataset_config.keys()), "Mismatch between dataset functions and file configurations"

# List of supported datasets, derived from the keys of the dataset_config dictionary.
# This list is used for validation and quick reference.
supported_datasets = list(dataset_config.keys())


custom_transformations = {
    'N-Cars': {
        'scale': (0.9, 1),
        'angle' : 10,
        'translate' : (0.05, 0.05),
        'hflip' : False,
    },
    'N-MNIST': {
        'scale': (0.9, 1),
        'angle' : 10,
        'translate' : (0.1, 0.1),
        'hflip' : False,
    },
    'N-Caltech101': {
        'scale': (0.9, 1),
        'angle' : 10,
        'translate' : (0.1, 0.1),
        'hflip' : False,
    },
    'CIFAR10-DVS': {
        'scale': (0.9, 1),
        'angle' : 10,
        'translate' : (0.1, 0.1),
        'hflip' : True,
    },
    'ASL-DVS': {
        'scale': (0.9, 1),
        'angle' : 10,
        'translate' : (0.1, 0.1),
        'flip' : False,
    },
    'DVS-Gesture': {
        'scale': (0.9, 1),
        'angle' : 10,
        'translate' : (0.1, 0.1),
        'flip' : False,
    },
    
    # Default transformations can be defined as an empty list or specific transformations that are commonly used.
}