# from scripts.data_processing.dat2mat import dat2mat
# from scripts.data_processing.process_events import generate_event_representation, preprocess_events_list
import os
# from scripts.datasets.Dataset import EventsDataset
from scripts.datasets.Dataset import *
from scripts.data_processing import *

# import cv2

class CIFAR10DVS(EventsDataset):
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
        
        self.dataset_name = 'CIFAR10-DVS'
        # original data's frame dimensions
        self.height = 128
        self.width = 128

        # number of classes
        self.num_classes = 10

        # label of each class
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

        # read events and rotate events by 90 degrees CW
        events_dict = dat2mat(events_file_path, rotate90=True) # possibly replace with .txt events file parser
        # events_list = open(events_file_path, 'r').readlines()

        if self.split == 'train':
            # apply polarity or temporal augmentation if enabled
            self.apply_events_augmentation(events_dict)

        # convert to desired representation (check prior codes)
        event_frame = generate_event_representation(events_dict, self.width, self.height, delta_t=self.delta_t, representation=self.event_rep, channels=self.channels)

        # fix frame rotation
        # event_frame = np.rot90(event_frame, k=1, axes=(1,0)).copy()
        # event_frame = cv2.rotate(event_frame, cv2.ROTATE_90_CLOCKWISE)
        
        return event_frame