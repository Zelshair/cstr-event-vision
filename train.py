"""
Training and Evaluation Script for Event-Based Classification/Recognition Models.

This script sets up and executes the training and evaluation pipeline for a specified event-based vision model. It covers the entire workflow from initializing datasets and dataloaders, setting up the model, optimizer, and loss function, through to training the model, saving the trained model, and evaluating its performance on a test dataset.

The script utilizes a configuration object (`Configs`) to define various training parameters such as the dataset, model architecture, augmentation strategies, and more. This modular approach allows for easy adjustments to the training setup and supports experimenting with different configurations.

Usage:
    - Define a configuration instance (`Configs`) with desired settings.
    - Call the `main` function with this configuration object.
    - The script will train the model according to the specified configurations, save the best model, plot training history, and evaluate the model on a test dataset.

Features:
    - Supports using cached datasets to speed up training.
    - Allows for comprehensive augmentation configuration.
    - Facilitates evaluation with detailed classification reports.
    - Configurable for various datasets, models, and training regimes.

Example:
    From the command line, run:
    ```
    python train.py
    ```
    Ensure `config.py` contains the desired training configurations and that necessary datasets and models are accessible as per the configuration.

Notes:
    - The script is designed to be modified as needed for specific training and evaluation requirements.
    - For detailed configuration options, refer to the documentation/comments in `config/config.py`.
"""

import torch
from utils import *
from config import *

# main function to train a model and evaluate it
def main(config, cached_dataset = None, cached_test_dataset= None):

    # set a fixed randomization seed
    set_seed(config.SEED)

    # generate train, val, test datasets
    train_dataset, val_dataset, test_dataset = set_dataset(config, cached_dataset, cached_test_dataset)

    # create dataloader for training, validation, and testing sets
    train_dataloader, val_dataloader, test_dataloader = set_dataloaders(config, train_dataset, val_dataset, test_dataset)

    # Get frame size
    size = (train_dataset.dataset.width, train_dataset.dataset.height) if config.keep_size else (config.frame_size, config.frame_size)

    # initialize classification model
    model = get_classification_model(config, train_dataset.dataset.num_classes, size)

    # print training configuration info
    print_configuration(config, train_dataset.dataset)

    # initialize our optimizer
    opt = set_optimizer(config, model)

    # Set scheduler if enabled
    scheduler = set_scheduler(config, opt)    

    # set loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # train model and get training history
    model, history = train_model(model, train_dataloader, val_dataloader, config, opt, loss_function, scheduler)

    # save best model to file
    save_model(config, model)

    # plot training history and save to file
    plot_train_hist(history, config, model)

    # Test trained model on the test set
    predictions = evaluate_model(config, model, test_dataloader)

    # generate classification report
    gen_classification_report(config, test_dataset.class_index, predictions, test_dataset.classes)

    # clear model / book-keeping / garbage collection (needed when benchmarking multiple models)
    del model

    # delete cached validation set if was cached during augmentation (for garbage collection)
    if config.augmentation and config.cache_dataset and config.cache_transforms:
        del val_dataset.dataset


if __name__ == "__main__":
    # Example usage of training and testing a model on a given dataset.

    # Set randomized training augmentation configuration as needed
    aug_conf = AugmentationConfig(spatial=True, temporal=True, polarity=True)

    # Set Training and Evaluation configuration
    config = Configs(dataset='N-Cars', 
                     event_rep='cstr', 
                     channels=3, 
                     classifier='mobilenetv2', 
                     cache_dataset=True, 
                     cache_transforms=True, 
                     cache_test_set=False, 
                     delta_t=0, 
                     balanced_splits=True, 
                     save_results=True, 
                     visualize_results=False, 
                     augmentation_config=aug_conf)

    # Start the Training and Evaluation process
    main(config)