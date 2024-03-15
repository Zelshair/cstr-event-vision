# from scripts.data_processing.load_ndata import read_ndataset
# from scripts.data_processing.process_events import preprocess_events_dict, preprocess_events_list
import os
# from scripts.datasets.Dataset import EventsDataset
from scripts.datasets.Dataset import *
from scripts.data_processing import *

class NMNIST(EventsDataset):
    def __init__(self, 
                 csv_file, 
                 root_dir, 
                 event_rep='cstr', 
                 channels=1, 
                 split='train', 
                 delta_t = 0, 
                 cache_dataset = False, 
                 transform=None, 
                 keep_size=False,
                 frame_size = (35, 35),
                 cache_transforms=False, 
                 augmentation_config=None, 
                 use_mp=True) -> None:

        # initialize parent dataset class object
        super().__init__(csv_file, root_dir, event_rep, channels, split, delta_t, cache_dataset, transform, keep_size, frame_size, cache_transforms, augmentation_config, use_mp)
        
        self.dataset_name = 'N-MNIST'
        # original data's frame dimensions
        self.height = 35 # data ranges from 0 - 34
        self.width = 35 # data ranges from 0 - 34

        # number of classes
        self.num_classes = 10

        # label of each class
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        assert len(self.classes) == self.num_classes

        # save the indices of the correct labels for each datapoint
        self.class_index = self.dataset_data['class_index']

        # Cache the dataset if enabled
        if self.cache_dataset:
            self.cache_data()

    def get_event_frame(self, index):
        # read events file path
        events_path = self.dataset_data['events_file_path'][index]

        # create path to file
        events_file_path = os.path.join(self.root_dir, events_path)

        # read events
        events_dict = read_ndataset(events_file_path) # possibly replace with .txt events file parser
        # events_list = open(events_file_path, 'r').readlines()
        
        if self.split == 'train':
            # apply polarity or temporal augmentation if enabled
            self.apply_events_augmentation(events_dict)

        # convert to desired representation (check prior codes)
        event_frame = generate_event_representation(events_dict, self.width, self.height, self.delta_t, representation=self.event_rep, channels=self.channels)
        # event_frame = preprocess_events_list(events_list, 0, self.width, self.height, representation=self.event_rep, channels=self.channels)

        return event_frame