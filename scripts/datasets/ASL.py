import scipy.io as sio
import os
from scripts.datasets.Dataset import *
from scripts.data_processing import *

class ASLDVS(EventsDataset):
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
                 frame_size = (240, 180),
                 cache_transforms = False, 
                 augmentation_config=None, 
                 use_mp=True) -> None:
        
        # initialize parent dataset class object
        super().__init__(csv_file, root_dir, event_rep, channels, split, delta_t, cache_dataset, transform, keep_size, frame_size, cache_transforms, augmentation_config, use_mp)

        self.dataset_name = 'ASL-DVS'
        # original data's frame dimensions
        self.height = 180
        self.width = 240

        # offsets used to flip events' coordinates both horizontally and vertically
        self.y_flip = self.height - 1
        self.x_flip = self.width - 1

        # number of classes
        self.num_classes = 24

        # label of each class
        self.classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

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

        # read events from .mat file
        events_dict = sio.loadmat(events_file_path, squeeze_me = True)

        # rename 'pol' key so the data would be compatible with this framework
        events_dict['p'] = events_dict.pop('pol')

        # this step is ignored for now to improve run-time dataloading speed (header would be ignored in the remaining code)
        # # remove extra keys
        # events_dict.pop('__header__', None)
        # events_dict.pop('__version__', None)
        # events_dict.pop('__globals__', None)

        # flip events vertically and horizontally
        # (verified by visualizing and comparing with the original paper's visualizations)
        events_dict['y'] = self.y_flip - events_dict['y']
        events_dict['x'] = self.x_flip - events_dict['x']

        if self.split == 'train':
            # apply polarity or temporal augmentation if enabled
            self.apply_events_augmentation(events_dict)    

        # convert to desired representation (check prior codes)
        event_frame = generate_event_representation(events_dict, self.width, self.height, delta_t=self.delta_t, representation=self.event_rep, channels=self.channels)
        
        return event_frame