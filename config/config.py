"""
Configuration Module for Event-Based Vision Classification Model Training.

This module defines the `Configs` class used to configure various aspects of model training, including
data preprocessing, augmentation, model selection, and training hyperparameters. It imports specific 
configuration classes for normalization parameters and augmentation settings to provide a comprehensive
setup for training across different datasets and model architectures.
"""

# import the necessary packages
import torch
import multiprocessing
from config.norm_params_config import NormParams
from config.augmentation_config import AugmentationConfig

class Configs():
    """
    A configuration class encapsulating settings for model training and evaluation.

    Provides structured management of training configurations, including data preprocessing,
    augmentation, model architecture, training hyperparameters, and output management. 
    Configurations cover device selection, dataset parameters, augmentation strategies, model
    details (including choice and pretraining), normalization, data caching, and various paths
    for outputs and caching.

    Key Attributes (Partial List):
        - dataset, event_rep, channels: Dataset and event representation settings.
        - classifier, pretrained_classifier: Classification model architecture and pretraining flag.
        - batch_size, frame_size, keep_size: Data loading and preprocessing parameters.
        - normalization, augmentation_config: Normalization and augmentation settings.
        - delta_t: Maximum time window for event data processing.
        - cache_dataset, cache_transforms, cache_test_set: Boolean flags to enable caching of training, testing sets and their transforms to accelerate training.
        - Numerous paths for outputs, models, and caching.
        - DEVICE: Computation device based on CUDA availability.

    Additional settings, such as optimizer choice, learning rate, epochs, and more, are also configurable.
    Detailed descriptions of all available settings are provided as comments within the class.
    """

    def __init__(self, dataset = 'N-MNIST', event_rep = 'cstr', channels = 3, classifier = 'mobilenetv3s', \
            pretrained_classifier = True, batch_size = 64, frame_size = 224, keep_size = False, save_results=False,\
            cache_dataset=False, cache_transforms=False, cache_test_set = False, normalization='ImageNet', \
            balanced_splits = False, delta_t = 0, augmentation_config=AugmentationConfig(), visualize_results=False):

        # set dataset for training/testing
        self.dataset = dataset

        # select event representation
        self.event_rep = event_rep

        # number of the event representation's input channels
        self.channels = channels

        # set classifier
        self.classifier = classifier

        # load pretrained weights?
        self.pretrained_classifier = pretrained_classifier        

        # set batch size
        self.batch_size = batch_size       

        # specify desired frame dimensions
        self.frame_size = frame_size

        # override frame size when using inceptionv3
        if self.classifier == 'inceptionv3' or self.classifier == 'inceptionv3_aux':
            self.frame_size = 299

        # keep original dataset size or use frame_size
        self.keep_size = keep_size

        # Save results flag
        self.save_results = save_results

        # Save predictions to file when utilizing DVS-Gesture (for post processing)
        if self.dataset == 'DVS-Gesture':
            self.save_preds = True
        else:
            self.save_preds = False

        # Cache dataset flag
        self.cache_dataset = cache_dataset

        # Cache transformed data flag
        self.cache_transforms = cache_transforms

        # Cache val set flag (True by default)
        self.cache_val_set = cache_dataset        

        # Cache test set flag
        self.cache_test_set = cache_test_set

        # set normalization type
        self.normalization = normalization

        # set normalization parameters
        self.norm_params = None

        # set delta_t (max time window duration) for a sample's events. 0 = No limit.
        self.delta_t = delta_t

        # specify ImageNet mean and standard deviation
        if self.normalization == 'ImageNet':
            self.norm_params = NormParams(dataset_name = 'ImageNet', channels=self.channels)
        
        elif self.normalization is None:
            self.norm_params = NormParams(dataset_name = None, channels=self.channels)
        
        elif self.normalization == 'DatasetNorm':
            self.norm_params = NormParams(dataset_name = self.dataset, event_rep=self.event_rep, channels=self.channels)
        
        # flag to enable balanced train/val splits based on the number of samples per class
        self.balanced_splits = balanced_splits

        # set augmentation configuration object
        self.augmentation_config = augmentation_config

        # check if event data augmentation is enabled
        self.augmentation = augmentation_config.is_enabled()

        # flag to enable test set visualization
        self.visualize_results = visualize_results

        # set scheduler {None, 'StepLR', 'Plateau', 'Plateau'}
        self.scheduler = None

        # set weight decay
        self.weight_decay = 0

        # set optimizer
        self.optimizer = 'adam'
        # self.optimizer = 'sgd

        # set mean and std
        self.MEAN = self.norm_params.mean
        self.STD = self.norm_params.std
        
        # [CONSTANTS]:

        # determine the device we will be using for inference
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # define the train and val splits
        self.TRAIN_SPLIT = 0.75
        self.VAL_SPLIT = 1 - self.TRAIN_SPLIT

        # Set seed for randomization
        self.SEED = 23

        # Set training and optimization parameters
        self.INIT_LR = 3e-4
        self.EPOCHS = 50
        self.EARLY_STOP_THRESH = 10

        # Set relative output paths
        self.OUTPUT_PATH = 'output/'
        self.CKPT_PATH = 'check_point/'
        self.MODELS_PATH = 'models/'
        self.HIST_PATH = 'training_history/'
        self.RESULTS_PATH = 'results/'
        self.CACHE_DIR = 'cache/'        
        
        # set number of CPU cores available in your system or less if desired (for multiprocessing)
        self.CPU_CORES = multiprocessing.cpu_count()

        # set number of workers for loading the training set at run-time (0 for cached dataset without temporal or polarity augmentations, 8/12/16 if not cached, ideally = number of CPU threads available)
        self.NUM_WORKERS = 0 if self.cache_dataset and not (self.augmentation_config.temporal or self.augmentation_config.polarity) else self.CPU_CORES

        # set number of workers for loading the test set at run-time (0 for cached dataset, 8/12/16 if not cached, ideally = number of CPU threads available)
        self.NUM_TEST_WORKERS = 0 if self.cache_test_set else self.CPU_CORES

        # flag to enable multiprocessing when caching a dataset before training
        self.USE_MP = True if self.CPU_CORES > 0 else False

        # Flag to enable caching to disk (recommended)
        self.CACHE_TO_DISK = True

        # set persitent workers if conditions are met to speed up the training process when not caching the samples
        self.persistent_train_workers = True if ((not self.cache_dataset or self.augmentation_config.temporal or self.augmentation_config.polarity) and self.USE_MP) else False
        self.persistent_val_workers = True if (not self.cache_val_set and self.USE_MP and self.NUM_WORKERS > 0) else False

        # set best model's name, checkpoint, model path
        self.set_best_model()

    # helper function to set best model file names, checkpoint path and final model path
    def set_best_model(self):
        best_model_name = 'best_model_pretrained' if self.pretrained_classifier else 'best_model'
        best_model_name += '_org_size' if self.keep_size else '_' + str(self.frame_size)
        best_model_name += '_' + str(self.delta_t) + 'ms' if self.delta_t > 0 else ''
    
        if self.augmentation:
            best_model_name += '_aug_' 
            if self.augmentation_config.spatial:
                best_model_name += 'S'
            if self.augmentation_config.temporal:
                best_model_name += 'T'
            if self.augmentation_config.polarity:
                best_model_name += 'P'                            

        best_model_name +=  '-'+ self.normalization if self.normalization is not None else '-NoNorm'
        self.best_model = '_'.join(list((self.dataset, self.event_rep, self.classifier, str(self.channels) + 'C', best_model_name + '.pth')))