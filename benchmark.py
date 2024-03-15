"""
Automated Benchmarking Script for Event-Based Classification/Recognition Models.

This script facilitates automated training and evaluation across a range of configurations to systematically benchmark the performance of different model architectures on various datasets. It supports specifying a wide array of configurations, including datasets, event representations, classifier types, and more, allowing for comprehensive comparative analysis.

Usage:
    Modify the `benchmark_main` function to include the desired configurations and run the script directly. The script iterates over the provided configurations, training and evaluating models accordingly, and performs cleanup to manage memory usage efficiently.

Example:
    python benchmark.py
"""

import torch
import gc
from train import main
from utils import *
from config.augmentation_config import AugmentationConfig


# Example usage of benchmark function
def benchmark_main():
    configurations = {
        'datasets': ['N-MNIST', 'N-Cars'],
        'event_reps': ['bin_event_frame', 'cstr'],
        'classifiers': ['resnet18', 'resnet50', 'mobilenetv2', 'mobilenetv3l', 'mobilenetv3s', 'inceptionv3', 'inceptionv3_aux'],
        'channels_list': [3],
        'keep_size_list': [False],
        'pretrained_weights': [True],
        'save_results': True,
        'cache_dataset': True,
        'cache_transforms': True,
        'cache_test_set': True,
        'normalization': 'ImageNet',
        'balanced_splits': True,
        'delta_t': 0,
        'augmentation_config': AugmentationConfig(temporal=True, polarity=True),
    }
    # Run benchmark with the provided configurations
    benchmark(configurations)


def benchmark(configurations):
    """
    Executes a series of training and evaluation runs based on a variety of configurations.

    Iterates over each combination of provided configurations, including datasets, event representations, classifier types, and preprocessing options, to train and evaluate models. Supports special handling for models requiring specific input dimensions (e.g., Inception models).

    Parameters:
        configurations (dict): A dictionary specifying configurations for the benchmarking process. Expected keys include:
            - 'datasets': List of datasets to use.
            - 'event_reps': List of event representation methods.
            - 'classifiers': List of classifier architectures, including special cases like Inception models.
            - 'channels_list': List specifying the number of channels for each event representation.
            - 'keep_size_list': List of boolean values indicating whether to keep the original dataset size.
            - 'pretrained_weights': List of boolean values indicating whether to use pretrained weights.
            - Additional configurations such as 'save_results', 'cache_dataset', 'cache_transforms', 'cache_test_set', 'normalization', 'balanced_splits', 'delta_t', and 'augmentation_config'.

    The function dynamically adjusts configurations for each run, handling dataset caching, model training and evaluation, and memory cleanup to facilitate seamless benchmarking across the specified scenarios.

    Returns:
        None. Results of benchmarking runs are typically saved to files or printed to the console based on the configurations.
    """

    # Create a list of inception classifiers (they require an input dimensions of 299x299 and support 3-channels only)
    inception_classifiers = list_inception_classifiers(configurations['classifiers'])

    for dataset in configurations['datasets']:
        # Check if dataset is supported then get its corresponding paths
        dataset_root, train_csv_path, test_csv_path = get_dataset_paths(dataset)

        for event_rep in configurations['event_reps']:
            for keep_size in configurations['keep_size_list']:
                for channels in configurations['channels_list']:
                    config = create_config(dataset=dataset, event_rep=event_rep, channels=channels, keep_size=keep_size, **configurations)
                    
                    # Load and Cache training dataset
                    cached_train_dataset = create_dataset(config, dataset_root, train_csv_path, 'train', set_transforms(config, train=True))
                    
                    # Load and Cache testing dataset
                    cached_test_dataset = create_dataset(config, dataset_root, test_csv_path, 'test', set_transforms(config, train=False))

                    for classifier in configurations['classifiers']:
                        for pretrained in configurations['pretrained_weights']:
                            config.classifier, config.pretrained_classifier = classifier, pretrained
                            train_and_evaluate_model(config, cached_train_dataset, cached_test_dataset)

                    # delete cached dataset
                    del cached_train_dataset
                    del cached_test_dataset
                    
                    if channels == 3:
                        for classifier in inception_classifiers:
                            for pretrained in configurations['pretrained_weights']:
                                config.classifier, config.pretrained_classifier = classifier, pretrained
                                train_and_evaluate_model(config)

def train_and_evaluate_model(config, train_dataset=None, test_dataset=None):
    main(config, train_dataset, test_dataset)
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    benchmark_main()