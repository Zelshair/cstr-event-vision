'''
Python version of the matlab function Read_Ndataset.m for reading N-MNIST and N-Caltech, developed by Zaid El Shair.

Event data files are binary files (.bin) with the following format:
Each example is a separate binary file consisting of a list of events. Each event occupies 40 bits arranged as described below:

bit 39 - 32: Xaddress (in pixels)
bit 31 - 24: Yaddress (in pixels)
bit 23: Polarity (0 for OFF, 1 for ON)
bit 22 - 0: Timestamp (in microseconds)
'''
import numpy as np

def read_ndataset(filename, polarity_mode="binary"):
    """
    This function reads a file of event data in the N-MNIST/N-Caltech101 dataset format as input and returns a dictionary of the event data with the following keys (ts, x, y, p)
    
    Parameters:
        filename: The .bin binary filename containing the event data for either N-MNIST or N-Caltech101 datasets
    
    Returns:
        dict: dictionary containing event data. Keys = "x", "y", "p", "ts"
    """
    # open file in read+binary mode
    with open(filename, "rb") as f:
        # read event stream file into a numpy array of uint8
        evt_stream = np.fromfile(f, dtype=np.uint8)

    # create a dict for the event data
    TD = {}    

    # extract each event's timestamp in microseconds from the stream (refer to official script)
    x = (evt_stream[2::5]).astype('uint32')
    y = (evt_stream[3::5]).astype('uint32')
    z = (evt_stream[4::5]).astype('uint32')

    TD['ts'] = (((x & 127) << 16)+ (y << 8) + z)
    
    # extract each event's pixel x address
    TD['x'] = evt_stream[0::5] + 1

    # extract each event's pixel y address
    TD['y'] = evt_stream[1::5] + 1

    # extract each event's polarity and convert to 0/1 or 1/2 (off/on) by shifting right 7 times (+1 for binary mode)
    TD['p'] = evt_stream[2::5] >> 7 if polarity_mode == 'binary' else evt_stream[2::5] >> 7 + 1

    return TD

# example main function to show an example using read_ndataset() function
def example():
    # read events from a .bin file
    file = 'samples/image_0045.bin'
    events_dict = read_ndataset(file)

    # print first 10 events
    print(events_dict['ts'][0:10])
    print(events_dict['x'][0:10])
    print(events_dict['y'][0:10])
    print(events_dict['p'][0:10])

if __name__ == '__main__':
    example()