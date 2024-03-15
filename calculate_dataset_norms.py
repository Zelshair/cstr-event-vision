"""
Calculate Dataset Normalization Parameters Script.

This script calculates the mean and standard deviation (std) for each channel of a specified dataset. These normalization parameters are crucial for preprocessing inputs in machine learning models, ensuring that data fed into the model has a consistent scale.

The script demonstrates how to use utility functions and the Configs object to load a dataset, create a DataLoader for efficient processing, and compute the dataset's normalization parameters. It prints the calculated mean and std values along with some additional dataset information.

The calculation process can be adjusted for different datasets, event representations, and channel configurations by modifying the corresponding settings within the script.

Usage:
    Run this script directly from the command line to calculate and print the normalization parameters for the specified dataset. Adjust the `dataset_name`, `event_rep`, `channels`, and other settings as needed for different datasets or configurations.

Example:
    ```
    python calculate_dataset_norms.py
    ```

Note:
    - Ensure that the specified dataset is supported and correctly configured in the script.
    - The script is set to use the original dataset frame size without applying any transforms. Modify the `transform` variable if preprocessing is required before calculating norms.
    - The DataLoader is configured to work with multiple parallel workers to expedite the processing. Adjust `num_workers` in the Configs object as needed based on your system's capabilities.
"""


from torch.utils.data.dataloader import DataLoader
import time
from utils import *
from config.config import Configs

# Example usage of the function get_norm_params of a supported dataset
def main():
    # set dataset name
    dataset_name = 'N-Cars'

    # specify event-representation type
    event_rep = 'cstr'

    # set number of channels for the event-representation
    channels = 3

    # set transforms to None
    transform = None

    # Use original dataset frame size
    keep_size = True

    # Check if dataset is supported then get the required paths
    dataset_root, train_csv_path, _ = get_dataset_paths(dataset_name)

    config = Configs(dataset=dataset_name, event_rep=event_rep, channels=channels, keep_size = keep_size)

    # Load training dataset
    train_dataset = create_dataset(config, dataset_root, train_csv_path, 'train', transform)

    # Create a dataloader (to parse the full dataset much quicker with multiple parallel workers)
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=config.NUM_WORKERS, worker_init_fn=worker_init_fn)

    start_time = time.time()

    # find mean and std per each channel of the dataset's input
    mean, std = get_norm_params(dataloader)

    end_time = time.time()

    print('Total time taken: {:.2f} seconds\n'.format(end_time-start_time))
    print('dataset name:', train_dataset)
    print('dataset length:', len(train_dataset))
    print('event representation:', event_rep)
    print('number of channels:', channels)
    print("mean =", mean)
    print("std =", std)

if __name__ == "__main__":
    main()