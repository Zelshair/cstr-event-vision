from scripts.datasets import *
from config.dataset_config import *
import glob
import os
import copy
import random 
import torch

from torch.utils.data import random_split, Subset
from torchvision import transforms


# helper function to set dataset parameters and generate dataset objects
def set_dataset(config, cached_dataset = None, cached_test_dataset = None, VERBOSE = True):
    """
    Loads the specified event-based dataset from file, applies necessary transformations,
    and optionally caches it for efficient loading.

    Parameters:
    - config: Configuration object containing dataset and augmentation settings.
    - cached_dataset: Optionally, a preloaded training dataset object to use.
    - cached_test_dataset: Optionally, a preloaded testing dataset object to use.
    - VERBOSE: Boolean flag indicating whether to print detailed logs.

    Returns:
    - train_dataset: The training dataset, potentially split and transformed.
    - val_dataset: The validation dataset, potentially split and transformed.
    - test_dataset: The testing dataset, loaded and transformed.
    """

    # Check if dataset is supported then get the required paths
    dataset_root, train_csv_path, test_csv_path = get_dataset_paths(config.dataset)

    # Check if training dataset is already cached
    if cached_dataset:
        main_train_dataset = cached_dataset

    # Else, load the training dataset
    else:
        main_train_dataset = create_dataset(config, dataset_root, train_csv_path, 'train', transform=set_transforms(config, True))

    if VERBOSE:
        # Print training dataset configuration
        main_train_dataset.print_dataset()

    # Splitting training dataset into training and validation sets
    if config.balanced_splits:
        # Split training dataset uniformly into train/val splits based on each class
        train_dataset, val_dataset = get_balanced_splits(main_train_dataset, config, VERBOSE)
    else:
        # Split training dataset randomly into train/val splits
        train_dataset, val_dataset = get_random_splits(main_train_dataset, config)

    # Handle transformations and caching for the validation dataset
    if config.augmentation:
        if config.cache_transforms and VERBOSE:
            print("[INFO] caching the validation set")
        
        # Create a copy of the original dataset
        val_dataset.dataset = copy.deepcopy(val_dataset.dataset)
        
        # Set split to 'val'
        val_dataset.dataset.split = 'val'
        val_dataset.dataset.transform = set_transforms(config, False)

        # Cache validation set if required
        val_dataset.dataset.cache_dataset = config.cache_dataset
        val_dataset.dataset.cache_transforms = config.cache_transforms
        val_dataset.dataset.use_mp = False if val_dataset.dataset.dataset_cached else True
        val_dataset.dataset.cache_val_samples(val_dataset.indices)

    # Load or use cached testing dataset
    if cached_test_dataset:
        test_dataset = cached_test_dataset
    else:
        test_dataset = create_dataset(config, dataset_root, test_csv_path, 'test', transform=set_transforms(config, False))

    return train_dataset, val_dataset, test_dataset

# Helper function to generate the specified dataset-type object
def create_dataset(config, dataset_root, csv_path, split = 'train', transform = None):
    # Determine caching based on the split and config
    cache_dataset = False
    cache_transforms = False

    if split in ['train', 'val', 'test']:
        cache_dataset = config.cache_dataset
        if split == 'train':
            # cache dataset only if flag is enabled and temporal and polarity augmentations are disabled
            cache_dataset = cache_dataset and not (config.augmentation_config.temporal or config.augmentation_config.polarity)
            # do not cache transformations if runtime training augmentations are enabled
            cache_transforms = config.cache_transforms and not config.augmentation
        
        elif split == 'val':
            cache_dataset = config.cache_val_set
            cache_transforms = config.cache_transforms 
        
        elif split == 'test':
            cache_dataset = config.cache_test_set
            cache_transforms = config.cache_test_set

        else:  # for other future splits
            cache_transforms = config.cache_transforms    

    # Lookup and instantiate the dataset if supported
    if config.dataset in supported_datasets:
        dataset = dataset_functions[config.dataset](csv_path, dataset_root, config.event_rep, config.channels, split, config.delta_t, cache_dataset, transform, config.keep_size, (config.frame_size, config.frame_size), cache_transforms, config.augmentation_config, config.USE_MP)
    else:
        raise TypeError(f"Dataset {config.dataset} is not supported!")
    
    return dataset


# Helper function to check if dataset is supported and get paths to dataset files
def get_dataset_paths(dataset_name):
    """
    Retrieves the root directory and CSV file paths for a specified dataset.

    This function checks if the given dataset is supported, based on its name, and then constructs
    the paths to the training and testing CSV files based on the dataset's configuration. It simplifies
    access to dataset-specific paths, ensuring consistency and reducing redundancy in path construction
    across different parts of the application.

    Parameters:
    - dataset_name (str): The name of the dataset for which paths are being retrieved. This name
      should correspond to one of the keys in the global `supported_datasets` list.

    Returns:
    - tuple: A tuple containing three strings:
        - The root directory path of the dataset (`dataset_root`).
        - The full path to the training CSV file (`train_csv_path`).
        - The full path to the testing CSV file (`test_csv_path`).

    Raises:
    - ValueError: If the `dataset_name` is not recognized as a supported dataset. This ensures
      that the function only processes valid datasets as defined in the project's configuration.
    """
    if dataset_name not in supported_datasets:
        raise ValueError(f"Dataset {dataset_name} is not supported in the configuration.")

    dataset_root = dataset_config[dataset_name]['root']
    train_csv_file = dataset_config[dataset_name]['train_csv']
    test_csv_file = dataset_config[dataset_name]['test_csv']

    train_csv_path = os.path.join(dataset_root, train_csv_file)
    test_csv_path = os.path.join(dataset_root, test_csv_file)

    return dataset_root, train_csv_path, test_csv_path


# Helper function to get balanced splits for train/val
def get_balanced_splits(dataset, config, VERBOSE=True):
    ''' This function splits the dataset into 2 balanced subsets for each class and split
    '''
    # create dict of lists of indices samples of each class
    class_indices = {i: [] for i in range(dataset.num_classes)}
    # generate uniformly split sets based 

    # loop over dataset samples and split into sets based on each sample's class index
    for index in range(len(dataset)):
        # class_indices[dataset[index][1]].append(index)
        class_indices[dataset.dataset_data['class_index'][index]].append(index)
        if VERBOSE:
            print('balancing dataset sample [{}/{}]'.format(index+1, len(dataset)), end='\r')
    
    if VERBOSE:
        print()

    # create lists of indices for each split
    train_subset_indices = []
    val_subset_indices = []

    for i in range(dataset.num_classes):
        # calculate the train/validation split samples per class
        num_train_samples = int(len(class_indices[i]) * config.TRAIN_SPLIT)

        # randomly shuffle indices of this class
        random.shuffle(class_indices[i])

        # add each split's indices of the current class of to their lists
        train_subset_indices.extend(class_indices[i][:num_train_samples])
        val_subset_indices.extend(class_indices[i][num_train_samples:])

    # create a train and a val subset for this class based on the generated indices
    train_dataset = Subset(dataset, train_subset_indices)
    val_dataset = Subset(dataset, val_subset_indices)

    return train_dataset, val_dataset

# Helper function to get random splits for train/val
def get_random_splits(dataset, config):
    ''' This function splits the dataset into 2 randomly sorted subsets
    '''    
    # Calculate the total train/validation split samples
    num_train_samples = int(len(dataset) * config.TRAIN_SPLIT)
    num_val_samples = len(dataset) - num_train_samples

    # Generate random train/val subsets based on the desired overall train/val split percentages
    (train_dataset, val_dataset) = random_split(dataset, [num_train_samples, num_val_samples], \
        generator=torch.Generator().manual_seed(config.SEED))

    return train_dataset, val_dataset

def get_default_transforms(config, augment=False):
    """Return default spatial augmentation transformations."""
    transform_list = [transforms.ToTensor()]

    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(config.frame_size, scale=(0.9, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(15, (0.2, 0.2)),
        ])

    return transform_list

def get_custom_transforms(size, scale, angle, translate, hflip):
    """Function to create a list of custom spatial augmentation transformations."""
    transforms_list = []

    if hflip:
        transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.extend([
        transforms.RandomResizedCrop(size, scale=scale),  # Example fixed size, adjust as necessary
        transforms.RandomAffine(angle, translate)
    ])

    return transforms_list

def set_transforms(config, train=True):
    """
    Configures and returns a composed torchvision transformation based on dataset specifications and augmentation settings.

    The function dynamically constructs a transformation pipeline that includes basic tensor conversion, optional spatial augmentations (e.g., random crops, flips, and affine transformations), resizing (if required), and normalization. Spatial augmentations and resizing are applied conditionally, based on the training mode and configuration settings.

    Parameters:
    - config (object): A configuration object that must include:
        - channels (int): Number of image channels. Must be positive.
        - frame_size (int): The target size for resizing operations.
        - augmentation_config (object): Specifies augmentation settings, including whether spatial augmentations are enabled.
        - dataset (str): Name of the dataset, used to select dataset-specific augmentations.
        - normalization (bool): Indicates whether normalization should be applied.
        - keep_size (bool): Indicates whether the original image size should be maintained.
        - MEAN (tuple): Mean values for normalization, applicable if normalization is enabled.
        - STD (tuple): Standard deviation values for normalization, applicable if normalization is enabled.
    - train (bool, optional): Indicates whether the transformation is being set for training data. Defaults to True, affecting whether augmentations are applied.

    Returns:
    - A torchvision.transforms.Compose object representing the configured transformation pipeline.

    Raises:
    - AssertionError: If the number of channels specified in the config object is not positive.
    """

    assert config.channels > 0, "Number of channels must be positive."
    transforms_list = [transforms.ToTensor()]

    # If augmenting spatially, add dataset-specific spatial augmentations before normalization for training
    if train and config.augmentation_config.spatial:
        if config.dataset in custom_transformations.keys():
            transform_params = custom_transformations[config.dataset]
            dataset_transforms = get_custom_transforms(
                config.frame_size, transform_params['scale'], transform_params['angle'],
                transform_params['translate'], transform_params['hflip']
                )
        else:
            dataset_transforms = get_default_transforms(config, augment=True)
        
        transforms_list.extend(dataset_transforms)

    # Else check if resizing is required if not training or spatial augmentations are not utilized
    else:
        if not config.keep_size:
            # Add Resize if spatial augmentations are not utilized and custom frame size is specified
            transforms_list.append(transforms.Resize((config.frame_size, config.frame_size)))
            # transforms_list.append(transforms.Resize(config.frame_size)) # use this instead to maintain aspect ratio

    # Add normalization transformation if required
    if config.normalization and config.channels <= 3:
        transforms_list.append(transforms.Normalize(config.MEAN, config.STD))

    return transforms.Compose(transforms_list)

# function to get normalization parameters for specified dataset, event-rep for 1,2,3 channels
def get_norm_params(dataset, VERBOSE=True):
    # Initialize the accumulators for mean and standard deviation
    mean = 0.0
    std = 0.0

    dataset_size = len(dataset)

    # verify the batch size is 1 if a dataloader is provided
    if type(dataset) == torch.utils.data.dataloader.DataLoader:
        assert dataset.batch_size == 1, "Warning: when using a dataloader the batch size must be set to 1 to get correct measurements."


    # Compute the mean and standard deviation over the entire dataset
    for i, (frame, _) in enumerate(dataset):
        if VERBOSE:
            print("processing step [{}/{}]".format(i+1, dataset_size), end='\r')

        # check if a dataloader was provided (data is provided in batches of 1)
        if type(dataset) == torch.utils.data.dataloader.DataLoader:
            # Accumulate the mean
            mean += frame.mean(dim=(0, 2, 3))
            
            # Accumulate the standard deviation
            std += frame.std(dim=(0, 2, 3))
        
        # otherwise assume a regular torch dataset is provided (assumes <channel, H, W> order)
        else: 
            # Accumulate the mean
            mean += frame.mean(dim=(1, 2))
            
            # Accumulate the standard deviation
            std += frame.std(dim=(1, 2))
    
    if VERBOSE:
        print()

    # Divide by the number of frames to find the mean and standard deviation
    mean /= len(dataset)
    std /= len(dataset)

    # Convert the mean and standard deviation to numpy arrays
    mean = mean.numpy()
    std = std.numpy()

    return mean, std

def extract_image_list(path_list):
    image_list = []
    for path in path_list:
        image_list.append(path.split('\\')[-1].split('.')[0])
    return image_list

def save_events_to_file(BASE_PATH, file_name, list):
    csv_file = open(BASE_PATH + '/' + str(file_name) + '.csv', 'w')
    for line in list:
        csv_file.write(line + '\n')
    csv_file.close()

# helper function to get paths to all event files based on dataset format
def get_event_files(dataset_path, dataset_type, split=''):
    # check event-based dataset type:

    if split != '':
        # update path to point to the split folder only
        dataset_path = os.path.join(dataset_path, split, '')
        # convert to Linux dir format
        dataset_path = dataset_path.replace("\\", '/')        

    # 1) N-MNIST or N-Caltech101
    if dataset_type == "N-MNIST" or dataset_type=="N-Caltech101" or dataset_type==".bin":
        event_files = glob(dataset_path + '**/*.bin', recursive=True)
    
    # 2) N-Cars
    elif dataset_type == "N-Cars" or dataset_type == ".dat":
        event_files = glob(dataset_path + '**/*.dat', recursive=True)

    # 3) CIFAR10-DVS
    elif dataset_type == "CIFAR10-DVS" or dataset_type == "DVS-Gesture" or dataset_type == ".aedat":
        event_files = glob(dataset_path + '**/*.aedat', recursive=True)

    # 4) ASL-DVS
    elif dataset_type == "ASL-DVS" or dataset_type == ".mat":
        event_files = glob(dataset_path + '**/*.mat', recursive=True)

    # 5) custom dataset that uses .txt file for event data
    elif dataset_type == ".txt":
        event_files = glob(dataset_path + '**/*.txt', recursive=True)

    # 6) custom dataset that uses .csv file for event data
    elif dataset_type == ".csv":
        event_files = glob(dataset_path + '**/*.csv', recursive=True)

    # dataset not supported
    else:
        raise ValueError("dataset type [{}] not supported!".format(dataset_type))
    
    return event_files

# function to get dataset name (avoids issues due to extra slashes '/')
def get_dataset_name(dataset_path, data_split_flag=False):
    # list directories and files within datasetpath
    sample_folder_path = glob(os.path.join(dataset_path, "*"))[0]

    # convert to Linux dir format
    sample_folder_path = sample_folder_path.replace("\\", '/')

    # extract dataset name
    if data_split_flag == False:
        dataset_name = sample_folder_path.split('/')[-2]
    elif data_split_flag == True:
        dataset_name = sample_folder_path.split('/')[-3]

    return dataset_name