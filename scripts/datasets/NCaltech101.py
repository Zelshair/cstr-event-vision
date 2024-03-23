import os
from scripts.datasets.Dataset import *
from scripts.data_processing import *

class NCaltech101(EventsDataset):
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
                 cache_transforms=False, 
                 augmentation_config=None, 
                 use_mp=True) -> None:

        # initialize parent dataset class object
        super().__init__(csv_file, root_dir, event_rep, channels, split, delta_t, cache_dataset, transform, keep_size, frame_size, cache_transforms, augmentation_config, use_mp)
        
        self.dataset_name = 'N-Caltech101'
        # original data's frame dimensions
        self.height = 180 # resolution on average. each image is different but per-image resolution is not provided
        self.width = 240 #        

        # number of classes
        self.num_classes = 101

        # label of each class
        self.classes = ['accordion', 'airplanes', 'anchor', 'ant', 'BACKGROUND_Google', 'barrel', 'bass', 
            'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 
            'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 
            'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 
            'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'Faces_easy', 'ferry', 'flamingo', 
            'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 
            'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 
            'Leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'Motorbikes', 
            'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 
            'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 
            'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 
            'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

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
        events_dict = read_ndataset(events_file_path)

        if self.split == 'train':
            # apply polarity or temporal augmentation if enabled
            self.apply_events_augmentation(events_dict)  

        # convert to desired representation
        event_frame = generate_event_representation(events_dict, self.width, self.height, self.delta_t, representation=self.event_rep, channels=self.channels)

        return event_frame