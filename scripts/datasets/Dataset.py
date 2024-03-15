from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import time
import psutil
import multiprocessing
from config.config import Configs
import random
import bisect
import os
import pickle
import shutil

# class object to load an event-based dataset for classification
class EventsDataset(Dataset):
    def __init__(self, 
                 csv_file, 
                 root_dir, 
                 event_rep='cstr', 
                 channels=3, 
                 split='train', 
                 delta_t = 0, 
                 cache_dataset=False, 
                 transform=None, 
                 keep_size=True, 
                 frame_size = (224, 224), 
                 cache_transforms=False,
                 augmentation_config = None,
                 use_mp = True) -> None:

        super().__init__()

        self.dataset_data = pd.read_csv(csv_file) if csv_file != '' else None
        self.root_dir = root_dir
        self.split = split
        self.event_rep = event_rep
        self.channels = channels
        self.delta_t = delta_t
        self.cache_dataset = cache_dataset
        self.cache_transforms = cache_transforms
        self.use_mp = use_mp

        # initialize transformations
        # by default must convert to tensor
        if transform == None:
            self.transform = transforms.ToTensor()
            self.keep_size = True
        else:
            self.transform = transform
            self.keep_size = keep_size

        # set desired frame size for resizing prior to training and evaluation
        self.frame_size = frame_size            

        # to be set by the inheriting dataset type
        self.height = None
        self.width = None
        self.num_classes = None
        self.classes = None
        self.class_index = None
        self.dataset_name = None

        # initialize lists for cached input and labels
        self.cached_labels = []
        self.cached_event_frames = []
        self.cached_transformed_event_frames = []                

        # internal data members to record caching state
        self.dataset_cached = False
        self.transforms_cached = False

        # set augmentations configuration
        self.augmentation_config = augmentation_config
        
        # set cache directory
        self.cache_directory = Configs().CACHE_DIR        

    def __len__(self):
        return len(self.dataset_data)

    def __getitem__(self, index):
        # load cached dataset if enabled
        if self.cache_dataset:
            label = self.cached_labels[index]
            # load cached transformed data
            if self.cache_transforms:
                event_frame = self.cached_transformed_event_frames[index]

            # load cached data then apply transformations if cache_transforms was not enabled
            else: 
                event_frame = self.cached_event_frames[index]
                
                # apply required transformations if cache_transforms was not enabled
                # if self.cache_transforms is False:
                if self.transform:
                    event_frame = self.transform(event_frame)

        else:
            # get classification label
            label = self.dataset_data['class_index'][index]
            # get event_frame in the specified event_represenation and transformations
            event_frame = self.get_event_frame(index)
            
            # apply transformations
            if self.transform:
                event_frame = self.transform(event_frame)            

        return event_frame, label

    def cache_data(self):
        if self.cache_dataset:
            # get dataset size
            dataset_size = len(self.dataset_data)

            # check if dataset already cached
            if self.dataset_cached:
                # check if transformations are already cached or if caching transformations is not required
                if self.transforms_cached or self.cache_transforms is False:
                    return

                # cache transformations
                elif self.cache_transforms:
                    self.cache_data_transforms()

            else:
                # get total memory available
                available_mem = psutil.virtual_memory()[1]
                print("[INFO] Total memory (RAM) available = {:,} bytes".format(available_mem))
                      
                # calculate an approximate of the dataset size after transformation (with/without resizing)    
                # if (self.keep_size and len(self.transform.transforms) > 1) or not self.cache_transforms:
                total_mem_needed = dataset_size * 4 * self.height * self.width * self.channels
                print("[INFO] aproximate memory required to cache dataset = Number of Dataset samples({}) * H({}) * W({}) * C({}) * size of float.32 (4 bytes)".\
                    format(dataset_size, self.height, self.width, self.channels))
                print("[INFO] total memory required = {:,} bytes".format(total_mem_needed))

                if total_mem_needed > available_mem:
                    print("[WARNING] Insufficient memory available!")
                    print('[INFO] disabling dataset caching')
                    self.cache_dataset = False
                    return

                # Cache dataset
                cache_start_time = time.time()

                # cache dataset with or without multiprocessing
                self.cache_dataset_samples()

                cache_end_time = time.time()
                print("total time taken to cache dataset = {:.1f} seconds".format(cache_end_time - cache_start_time))

                # cache transformations
                if self.cache_transforms:
                    self.cache_data_transforms()
        else:
            return

    # function to cache dataset with or without transformations in a single or multiple processes
    def cache_dataset_samples(self):
        
        dataset_size = len(self)

        # use multiprocessing to load data if enabled
        if self.use_mp:
            if not self.dataset_cached:
                print(f"Caching dataset ({self.split} set) using multi-processing ({Configs().NUM_WORKERS} workers) ...")

                # Create cache directory if it doesn't exist
                os.makedirs(self.cache_directory, exist_ok=True)

                # Disk Caching
                if Configs().CACHE_TO_DISK:
                    with multiprocessing.Pool(Configs().NUM_WORKERS, initializer=self.init_worker, initargs=(Configs().NUM_WORKERS,)) as pool:
                        pool.map(self.worker_cache_data, range(dataset_size))                    

                    # Load cached data from disk to memory
                    self.load_cached_data(range(dataset_size))

                    # Delete cache directory after loading data into memory
                    self.delete_cache_directory()

                # Memory Caching
                else:                
                    with multiprocessing.Pool(Configs().NUM_WORKERS, initializer=self.init_worker, initargs=(Configs().NUM_WORKERS,)) as pool:
                        cached_data = pool.map(self.worker_cache_data, range(dataset_size)) # init_worker is needed if processes are not properly disrtibuted to cpu cores/threads

                    # with multiprocessing.Pool(Configs().NUM_WORKERS) as pool:
                    #     cached_data = pool.map(self.worker_cache_data, [i for i in range(dataset_size)])
                    
                    # unpack data
                    for i in range(len(cached_data)):
                        cached_sample = cached_data.pop()
                        self.cached_labels.append(cached_sample[0])
                        self.cached_event_frames.append(cached_sample[1])
                    
                    del cached_data

                    # correct the order of the list
                    self.cached_labels.reverse()
                    self.cached_event_frames.reverse()

        # Otherwise read data in the Main Thread
        else:
            if self.dataset_cached is False:
                for i in range(dataset_size):
                    # need to add check for the dataset size... find approximate size... ask user to verify if enough memory is available.
                    # 8 * C => Memory = 8 *m * n * p Bytes. or 8 * H * W * C
                    print("Caching dataset: [{}|{}]".format(str(i+1), dataset_size), end='\r')
                    # cache label by appending it to list
                    self.cached_labels.append(self.dataset_data['class_index'][i])
                    # generate the event frame in the specified event-representation
                    event_frame = self.get_event_frame(i)
                    
                    # append to event_frame to cached list
                    self.cached_event_frames.append(event_frame)

        # set flags
        self.dataset_cached = True

    def save_to_disk(self, index, data):
        filename = os.path.join(self.cache_directory, f"{index}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def load_from_disk(self, index):
        filename = os.path.join(self.cache_directory, f"{index}.pkl")
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def load_cached_data(self, indices):
        """
        params:
        - indices: an iterable range/list of indices for the function to parse through
        """

        # load cached data from file
        for i in indices:
            print(f"Caching sample [{i+1}|{len(indices)}] to memory", end='\r')

            # Load cached sample (i) from disk
            cached_sample = self.load_from_disk(i)

            # Read processed labels
            self.cached_labels.append(cached_sample['label'])

            # Read processed event-frame representation
            self.cached_event_frames.append(cached_sample['event_frame'])


    def load_cached_val_data(self, indices):
        """
        params:
        indices: an iterable range/list of indices for the function to parse through"""
        # load cached data from file
        for i, idx in enumerate(indices):
            print(f"Caching validation sample [{i+1}|{len(indices)}] to memory", end='\r')

            cached_sample = self.load_from_disk(idx)

            # Read processed labels
            self.cached_labels[idx] = cached_sample['label']

            # Read processed event-frame representation
            self.cached_event_frames[idx] = cached_sample['event_frame']

    def delete_cache_directory(self):
        if os.path.exists(self.cache_directory):
            shutil.rmtree(self.cache_directory)
            print("Cache directory deleted after loading data into memory.")

    # function to cache a dataset's transformations. Requires the original dataset to be cached beforehand.
    def cache_data_transforms(self):
        # check if dataset already cached
        if self.dataset_cached:
            # check if transformations are already cached or if caching transformations is not required
            if self.transforms_cached or self.cache_transforms is False:
                return
            # cache transformation for each datapoint
            else:
                # get total memory available
                available_mem = psutil.virtual_memory()[1]
                print("[INFO] Total memory (RAM) available = {:,} bytes".format(available_mem))
                           
                # get dataset length
                dataset_size = len(self)

                # verify that the 2nd transformation is a Resize() transformation
                assert type(self.transform.transforms[1]) == transforms.Resize

                # get the specified transformation size
                tranformed_size = self.transform.transforms[1].size[0]

                # calculate an estimate of the total memory needed to cache transforms
                total_mem_needed = dataset_size * 4 * tranformed_size * tranformed_size * self.channels
                
                print("[INFO] aproximate memory required to cache dataset with transforms = Number of Dataset samples({}) * H({}) * W({}) * C({}) * size of float.32 (4 bytes)".\
                    format(dataset_size, tranformed_size, tranformed_size, self.channels))
                
                # calculate total memory used for caching so far
                total_mem_cached = dataset_size * 4 * self.height * self.width * self.channels
                
                print("[INFO] total memory already in use by the cached dataset = {:,} bytes".format(total_mem_cached))
                print("[INFO] total memory required to cache transformed dataset = {:,} bytes".format(total_mem_needed))                
                print("[INFO] total extra memory required = {:,} bytes".format(total_mem_needed - total_mem_cached))

                # check if there is enough extra memory to cache the data transformations as well
                if total_mem_needed - total_mem_cached > available_mem:
                    print("[WARNING] Insufficient memory available! Please disable Transforms Caching option")
                    print('[INFO] proceeding with the original cached dataset without transformations')
                    self.cache_transforms = False
                    self.transforms_cached = False
                    return
                

                # Cache transformations in the main process (no multi-processing used here)
                cache_start_time = time.time()

                for i in range(dataset_size):
                    print("Caching dataset transformation: [{}|{}]".format(str(i+1), dataset_size), end='\r')

                    # apply transformation to the first available cached frame
                    transformed_event_frame = self.transform(self.cached_event_frames[0])

                    # remove the older non-transformed cached frame
                    del self.cached_event_frames[0]

                    # append transformed event frame to cache list
                    self.cached_transformed_event_frames.append(transformed_event_frame)

                # update flag
                self.transforms_cached = True

                cache_end_time = time.time()
                print("total time taken to cache transformations = {:.1f} seconds".format(cache_end_time - cache_start_time))   
        
        # Raise an error if the dataset was not cached prior to this function
        else:
            raise ValueError("Dataset transforms were not cached. Must cache dataset before attempting to cache the transforms!")   
         
    
    def cache_val_samples(self, val_indices: list):
        """
        Caches only the validation subset samples of a given dataset using the specified validation set indices.

        Args:
            val_indices (list): The indices pretaining to the validation set relative to the original dataset.
        """
        # sort indices after they were shuffled
        val_indices.sort()
        
        # check if dataset caching is enabled
        if self.cache_dataset:
            # get number of validation set samples
            subset_size = len(val_indices)

            # check if dataset already cached
            if self.dataset_cached:
                # check if transformations are already cached or if caching transformations is not required
                if self.transforms_cached or self.cache_transforms is False:
                    return

                # cache transformations
                elif self.cache_transforms:
                    # set cached list as dictionary instead
                    self.cached_transformed_event_frames = {}

                    # cache validation set samples transformations
                    self.cache_val_transforms(val_indices)

            else:
                # get total memory available
                available_mem = psutil.virtual_memory()[1]
                print("[INFO] Total memory (RAM) available = {:,} bytes".format(available_mem))
                      
                # calculate an approximate of the dataset size after transformation (with/without resizing)    
                total_mem_needed = subset_size * 4 * self.height * self.width * self.channels
                print("[INFO] aproximate memory required to cache dataset = Number of validation set samples({}) * H({}) * W({}) * C({}) * size of float.32 (4 bytes)".\
                    format(subset_size, self.height, self.width, self.channels))
                
                print("[INFO] total memory required = {:,} bytes".format(total_mem_needed))

                if total_mem_needed > available_mem:
                    print("[WARNING] Insufficient memory available!")
                    print('[INFO] disabling dataset caching')
                    self.cache_dataset = False
                    return
                
                # set cached lists as dictionaries instead
                self.cached_labels = {}
                self.cached_event_frames = {}

                # Record cache starting time
                cache_start_time = time.time()

                # cache dataset with or without multiprocessing
                if self.use_mp:
                    # Create cache directory if it doesn't exist
                    os.makedirs(self.cache_directory, exist_ok=True)

                    print("caching validation set using multiprocessing with {} workers...".format(Configs().NUM_WORKERS))

                    # Disk Caching
                    if Configs().CACHE_TO_DISK:
                        with multiprocessing.Pool(Configs().NUM_WORKERS, initializer=self.init_worker, initargs=(Configs().NUM_WORKERS,)) as pool:
                            pool.map(self.worker_cache_data, [i for i in val_indices])

                        # Load cached data from disk to memory
                        self.load_cached_val_data([i for i in val_indices])

                        # Delete cache directory after loading data into memory
                        self.delete_cache_directory()                        

                    # Memory Caching
                    else:
                        with multiprocessing.Pool(Configs().NUM_WORKERS, initializer=self.init_worker, initargs=(Configs().NUM_WORKERS,)) as pool:
                            cached_data = pool.map(self.worker_cache_data, [i for i in val_indices]) # init_worker is needed if processes are not properly disrtibuted to cpu cores/threads
                                                    
                        # with multiprocessing.Pool(Configs().NUM_WORKERS) as pool:
                        #     cached_data = pool.map(self.worker_cache_data, [i for i in val_indices])

                        assert len(cached_data) == len(val_indices)

                        # unpack data and append to cache dicts
                        for i in val_indices:
                            cached_sample = cached_data.pop()
                            self.cached_labels[i] = cached_sample[0]
                            self.cached_event_frames[i] = cached_sample[1]
                        
                        del cached_data
                    
                # Otherwise read and cache data in the MainThread
                else:
                    if self.dataset_cached is False:
                        for i in val_indices:
                            # need to add check for the dataset size... find approximate size... ask user to verify if enough memory is available.
                            # 8 * C => Memory = 8 *m * n * p Bytes. or 8 * H * W * C
                            print("Caching dataset: [{}|{}]".format(str(i+1), subset_size), end='\r')

                            # cache label by appending it to dict of val labels
                            self.cached_labels[i] = self.dataset_data['class_index'][i]

                            # generate the event frame in the specified event-representation
                            event_frame = self.get_event_frame(i)
                            
                            # append to event_frame to cached list
                            self.cached_event_frames[i] = event_frame

                # set flags
                self.dataset_cached = True

                # Record caching end time
                cache_end_time = time.time()
                print("total time taken to cache validation set samples = {:.1f} seconds".format(cache_end_time - cache_start_time))

                # cache val set transforms if enabled
                if self.cache_transforms:
                    # set cached list as dictionary instead
                    self.cached_transformed_event_frames = {}

                    # cache validation set transforms
                    self.cache_val_transforms(val_indices)

        else:
            return


    def cache_val_transforms(self, val_indices):
        # check if dataset already cached
        if self.dataset_cached:            
            # check if transformations are already cached or if caching transformations is not required
            if self.transforms_cached or self.cache_transforms is False:
                return
            
            # cache transformation for each datapoint
            else:
                # get total memory available
                available_mem = psutil.virtual_memory()[1]
                print("[INFO] Total memory (RAM) available = {:,} bytes".format(available_mem))
                        
                # get dataset length
                subset_size = len(val_indices)

                # verify that the 2nd transformation is a Resize() transformation
                assert type(self.transform.transforms[1]) == transforms.Resize

                # get the specified transformation size
                tranformed_size = self.frame_size[0]

                # calculate an estimate of the total memory needed to cache transforms
                total_mem_needed = subset_size * 4 * tranformed_size * tranformed_size * self.channels
                print("[INFO] aproximate memory required to cache validation set with transforms = Number of Dataset samples({}) * H({}) * W({}) * C({}) * size of float.32 (4 bytes)".\
                    format(subset_size, tranformed_size, tranformed_size, self.channels))
                total_mem_cached = subset_size * 4 * self.height * self.width * self.channels
                print("[INFO] total memory already in use by the cached validation set = {:,} bytes".format(total_mem_cached))
                print("[INFO] total memory required to cache transformed validation set = {:,} bytes".format(total_mem_needed))                
                print("[INFO] total extra memory required = {:,} bytes".format(total_mem_needed - total_mem_cached))

                # check if there is enough extra memory to cache the transforms as well
                if total_mem_needed - total_mem_cached > available_mem:
                    print("[WARNING] Insufficient memory available! Please disable Transforms Caching option")
                    print('[INFO] proceeding with the original cached dataset without transformations')
                    self.cache_transforms = False
                    self.transforms_cached = False
                    return

                # Cache transformations in the main process (no multi-processing used here)
                cache_start_time = time.time()

                for i, index in enumerate(val_indices):
                    print("Caching validation set transformations: [{}|{}]".format(str(i+1), subset_size), end='\r')

                    # apply transformation to the first available cached frame
                    transformed_event_frame = self.transform(self.cached_event_frames[index])

                    # # remove the older non-transformed cached frame
                    # del self.cached_event_frames[index]

                    # append transformed event frame to cache list
                    self.cached_transformed_event_frames[index] = transformed_event_frame
                
                # delete previously cached dataset used for the validation split
                del self.cached_event_frames

                # update flag
                self.transforms_cached = True

                cache_end_time = time.time()
                print("total time taken to cache validation set transformations = {:.1f} seconds".format(cache_end_time - cache_start_time))   
    
        # Raise an error if the dataset was not cached prior to this function
        else:
            raise ValueError("Validation set transforms were not cached. Must cache validation set before attempting to cache the transforms!")   

    # get dataset name
    def get_dataset_name(self):
        return self.dataset_name

    # function to print dataset parameters
    def print_dataset(self):
        # check if child dataset is not set.
        if self.dataset_name is None:
            raise NotImplementedError('Parent class should not be used directly. Must inherit this class instead.')
        else:
            print("\n[Event Dataset Information]:")
            print("\tDataset name:", self.dataset_name)
            print("\tNumber of classes = ", self.num_classes)
            print("\tClasses:", self.classes)
            print("\tOriginal dataset frame dimensions = {}x{}".format(self.width, self.height))
            print("\tTotal datapoints =", len(self.dataset_data))
            print("\tSplit =", self.split)
            print("\tCache dataset =", self.cache_dataset)
            print("\tCache transforms =", self.cache_transforms)
            print()

    # get dataset name
    def __str__(self):
        # check if child dataset is not set.
        if self.dataset_name is None:
            raise NotImplementedError('Parent class should not be used directly. Must inherit this class instead.')
        else:
            return self.dataset_name
        
    def get_num_items_class(self):
        # group the data by classes
        class_groups = self.dataset_data.groupby('class_index')
        print("class\tnumber of samples")
        for name, group in class_groups:
            print("{}\t{}".format(self.classes(int(name)), len(group)))

    def worker_cache_data(self, i):
        label = self.dataset_data['class_index'][i]
        event_frame = self.get_event_frame(i)

        # Disk-based Caching
        if Configs().CACHE_TO_DISK:
            # Compose the data structure
            data = {
                'label': label,
                'event_frame': event_frame
            }

            # Save to disk
            self.save_to_disk(i, data)            
        
        # Memory-based Caching
        else:
            return label, event_frame

    def worker_cache_transforms(self, i):
        if self.cache_transforms:
            event_frame = self.transform(self.cached_event_frames[i])
 
        return event_frame

    def init_worker(self, num_cores):
        """
        Initialize each worker by setting the CPU affinity based on its process ID.

        Args:
        - num_cores (int): Total number of cores available.
        """
        worker_id = os.getpid()  # Use the process ID as a unique identifier for the worker
        core_id = worker_id % num_cores
        p = psutil.Process(worker_id)
        p.cpu_affinity([core_id])

    # function to apply temporal or polarity augmentations to the sample's events
    def apply_events_augmentation(self, events_dict, polarity_mode = 'binary'):
        if self.augmentation_config is not None:
            # apply temporal augmentation by: shifting events by a random offset
            if self.augmentation_config.temporal:
                # check if delta_t is not specified (time window length)
                if self.delta_t == 0:
                    # assume time starts from 0 us
                    window_start_time = 0
                    
                    # read last event's timestamp
                    last_event_ts = events_dict['ts'][-1]
                    
                    # find delta_t
                    delta_t = last_event_ts - window_start_time

                # else if delta_t is specified (time window length)
                elif self.delta_t > 0:
                    delta_t = self.delta_t
                    # find last event's timestamp
                    last_event_ts = events_dict['ts'][-1]
                    # find the window start time using the specified delta_t value
                    window_start_time = last_event_ts - delta_t

                # raise error if delta_t is negative
                else:
                    raise ValueError("delta_t must not be negative. delta_t value = {}".format(self.delta_t))

                # generate a random number between -1 and 1
                random_value = (random.random() - 0.5) * 2

                # set max temporal threshold (shift between 0% - threshold% of the sample's duration)
                ts_threshold = self.augmentation_config.ts_threshold

                # verify that ts_threshold is between 0 and 1
                assert ts_threshold >= 0 and ts_threshold <= 1

                # calculate the random temporal offset between [- deltaT * ts_threshold : deltaT * ts_threshold]
                ts_offset = round(random_value  * ts_threshold * delta_t)

                # add offset to the events timestamps (note: limited to 32 bits)
                events_dict['ts'] = events_dict['ts'].astype('int32') + ts_offset

                # remove events outside the sample's original time range [0, delta_t]
                self.filter_events_ts(events_dict, delta_t, window_start_time)                        
            
            # apply polarity augmentation by: flipping all the events polarity before being converted to an intermediate represenation
            if self.augmentation_config.polarity:
                # set threshold for applying the polarity augmentation
                p_threshold = self.augmentation_config.p_threshold

                # generate a random number between 0 and 1
                random_value = random.random()

                # apply polarity flip if the random value is above the threshold
                if random_value > p_threshold:
                    if polarity_mode == 'binary':
                        # flip all events using xor
                        events_dict['p'] ^= 1

    # helper function to filter out with timestamps not within the range [0, delta_t]
    def filter_events_ts(self, events_dict, delta_t, window_start_time = 0):

        # use bisection method to find the index of the first event within the specified time window (<= window start time)
        start_index = bisect.bisect_left(events_dict['ts'], window_start_time)
        # use bisection method to find the index of the last event within the specified time window (>= delta_t)
        end_index = bisect.bisect_right(events_dict['ts'], delta_t)

        # remove events with timestamps not within the original time range
        events_dict['ts'] = events_dict['ts'][start_index:end_index] 
        events_dict['x'] = events_dict['x'][start_index:end_index] 
        events_dict['y'] = events_dict['y'][start_index:end_index] 
        events_dict['p'] = events_dict['p'][start_index:end_index] 

        assert len(events_dict['ts']) == len(events_dict['x']) == len(events_dict['y']) == len(events_dict['p'])

    # helper function to get original frame's dimensions
    def get_frame_dims(self):
        return self.width, self.height
    
    # helper function to get class names
    def get_class_names(self):
        return self.classes    