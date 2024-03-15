"""
Split DVS-Gesture Event Sequences Script.

This script is designed to split DVS-Gesture recorded event sequences into sub-sequences for each gesture performed within. Originally, sequences in the dataset have users performing multiple gestures consecutively. This script divides each sequence into individual files, each corresponding to a single gesture, facilitating gesture-wise analysis and training.

The DVS-Gesture dataset referenced is from the paper:
- A. Amir, B. Taba, D. Berg, T. Melano, J. McKinstry, C. Di Nolfo, et al., "A low power fully event-based gesture recognition system", Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 7243-7252, Jul. 2017.

Usage:
    Modify `input_dir` and `output_dir` to point to the dataset's location and where you want the split sequences to be saved, respectively. The script uses `train_list_file` and `test_list_file` to know which trials belong to the training and testing sets.

    This script should be run from the command line after setting the appropriate paths:
    ```
    python split_dvs_gesture_seqs.py
    ```

Features:
    - Automatically creates directories for each gesture class based on the labels provided in the dataset's label files.
    - Supports splitting both the training and testing sets as specified by separate lists of trials.
    - Utilizes efficient event file parsing and writing utilities to handle the DVS-Gesture's '.aedat' data format.

Note:
    - Ensure the DVS-Gesture dataset is correctly placed and accessible at the specified `input_dir`.
    - The script assumes the existence of `_labels.csv` files for each recorded sequence, detailing the start and end times of gestures.
"""


import os
import struct
import bisect

from scripts.data_processing.read_aedat import *
from scripts.data_processing.write_events_to_file import *

def parse_labels_file(filename):
    # Initialize list to store label data
    labels_list = []    
    # Open file and skip the first line (header)
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            # Split line by whitespace
            label_data = line.strip().split(',')

            # Extract class index, start time, and end time
            class_index = int(label_data[0])
            start_time = int(label_data[1])
            end_time = int(label_data[2])

            # Append data to label list
            labels_list.append((class_index, start_time, end_time))
    
    return labels_list

def parse_split_file(filename):
    # Initialize list to store label data
    samples_list = []    
    # Open file and skip the first line (header)
    with open(filename, 'r') as f:
        for line in f:
            if line != '':
                # Split line by . to get filename without extension type and append to list
                samples_list.append(line.strip().split('.')[0])
    return samples_list

def split_sample_events(events_dict, labels_list, output_dir):
    # Loop over each label and create a separate file for each label
    for label in labels_list:
        class_index, start_time, end_time = label
        class_dir = os.path.join(output_dir, f'{class_index}')

        # Create class directory if it does not exist
        if not os.path.exists(class_dir):
            os.makedirs(class_dir, exist_ok=True)

            class_sample_file = os.path.join(output_dir, f'class{class_index}.aedat')

            # Open output file and write header
            with open(class_sample_file, 'wb') as f:
                f.write(struct.pack('HHIIIII', 0, 0, 28, 0, 0, 0, 0))

                # Loop over event data and write events to output file
                for i in range(len(events_dict['ts'])):
                    timestamp = events_dict['ts'][i]
                    x = events_dict['x'][i]
                    y = events_dict['y'][i]
                    polarity = events_dict['p'][i]

                    # If the event falls within the label start and end times, write it to the output file
                    if timestamp >= start_time and timestamp <= end_time:
                        data = ((x & 0x00001FFF) << 17) | ((y & 0x00001FFF) << 2) | (polarity << 1)
                        f.write(struct.pack('II', data, timestamp))        

def split_dataset_samples(input_dir, train_list_file, test_list_file, output_dir='output/', VERBOSE=True):
    # set paths to split files
    train_list_path = os.path.join(input_dir, train_list_file)
    test_list_path = os.path.join(input_dir, test_list_file)

    # read training and testing split sample names
    train_list = parse_split_file(train_list_path)
    test_list = parse_split_file(test_list_path)

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create output directory for the training and testing split if it does not exist
    train_output_dir = os.path.join(output_dir, 'train')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir, exist_ok=True)

    test_output_dir = os.path.join(output_dir, 'test')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir, exist_ok=True)

    # split each set's samples and write to file according to each class
    if VERBOSE:
        print('Splitting the dataset train samples:')
    split_set_samples(train_list, train_output_dir)

    if VERBOSE:
        print('Splitting the dataset test samples')    
    split_set_samples(test_list, test_output_dir)    

def split_set_samples(samples_list, split_output_dir, VERBOSE=True):
    # Read every train sample and split per class
    for i, sample_name in enumerate(samples_list):
        if VERBOSE:
            print('splitting sample: [{}|{}]'.format(i+1, len(samples_list)), end='\r')
            
        sample_path = os.path.join(input_dir, sample_name)
        events_file = sample_path + '.aedat'
        if os.path.isfile(events_file):
            events_dict = read_aedat_file(events_file)

            labels_file = sample_path + '_labels.csv'
            labels_list = parse_labels_file(labels_file)

        # Loop over each label and create a separate file for each label
        for i, label in enumerate(labels_list):
            class_index, start_time, end_time = label
            class_dir = os.path.join(split_output_dir, str(class_index))

            # Create class directory if it does not exist
            if not os.path.exists(class_dir):
                os.makedirs(class_dir, exist_ok=True)

            class_sample_file = os.path.join(class_dir, sample_name + str(i) + '_' + str(class_index) + '.aedat')

            # use bisection method to find the index of the events within the specified sample time window
            start_index = bisect.bisect_left(events_dict['ts'], start_time)
            end_index = bisect.bisect_right(events_dict['ts'], end_time)

            # extract events within the sample time and add to dict of sample events
            sample_events_dict = {
                'x': events_dict['x'][start_index:end_index],
                'y': events_dict['y'][start_index:end_index],
                'ts': events_dict['ts'][start_index:end_index],
                'p': events_dict['p'][start_index:end_index]
            }

            # write extracted sample's specified class's events to file
            write_events_to_file(class_sample_file, sample_events_dict)
    
    # print new line
    if VERBOSE:
        print()

if __name__ == "__main__":
    # Define input and output directories
    input_dir = '../Datasets/DVS_Gesture/DvsGesture/'
    output_dir = '../Datasets/DVS_Gesture/DvsGesture/'

    # train split file
    train_list_file = 'trials_to_train.txt'

    # test split file
    test_list_file = 'trials_to_test.txt'

    # split the dataset's samples
    split_dataset_samples(input_dir, train_list_file, test_list_file, output_dir)