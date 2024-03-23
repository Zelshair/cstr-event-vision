import os
from scripts.datasets.Dataset import *
from scripts.data_processing import *

class NCars(EventsDataset):
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
                 frame_size = (128, 128),
                 cache_transforms=False, 
                 augmentation_config=None, 
                 use_mp=True) -> None:
        # initialize parent dataset class object
        super().__init__(csv_file, root_dir, event_rep, channels, split, delta_t, cache_dataset, transform, keep_size, frame_size, cache_transforms, augmentation_config, use_mp)
        
        self.dataset_name = 'N-Cars'
        # original data's frame dimensions
        # data was captured using ATIS. Supplemental material of N-Cars specifies this resolution processed
        self.height = 128
        self.width = 128

        # number of classes
        self.num_classes = 2

        # label of each class
        self.classes = ['background', 'cars']

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
        events_dict = load_atis_data(events_file_path)
        if self.split == 'train':
            # apply polarity or temporal augmentation if enabled
            self.apply_events_augmentation(events_dict)

        # convert to desired representation
        event_frame = generate_event_representation(events_dict, self.width, self.height, self.delta_t, representation=self.event_rep, channels=self.channels)

        return event_frame