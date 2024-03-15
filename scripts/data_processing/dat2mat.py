'''
Python version of the matlab function dat2mat.m for reading CIFAR10-DVS dataset

Event data files are binary files (.aedat) with the following format:
'UPDATE    Each example is a separate binary file consisting of a list of events. Each event occupies 40 bits arranged as described below:

    bit 39 - 32: Xaddress (in pixels)
    bit 31 - 24: Yaddress (in pixels)
    bit 23: Polarity (0 for OFF, 1 for ON)
    bit 22 - 0: Timestamp (in microseconds)'
'''
import numpy as np
import time

def dat2mat(file, rotate90 = False, polarity_mode="binary", debug=False):
    """
    This function reads a file of event data in the CIFAR10-DVS dataset format as input and returns a dictionary of the event data with the following keys (ts, x, y, p)
    
    Parameters:
        file: The path to the .aedat file containing the event data for either MNIST-DVS or CIFAR10-DVS datasets
        rotate90: flag to enable rotating the events by 90 degrees CW
    Returns:
        dict: dictionary containing event data. Keys = "x", "y", "p", "ts"
    """

    if debug:
        start_t = time.time()
    
    # set max events to avoid long file processing time
    max_events = int(1e6)

    # open file in read+binary mode
    with open(file, "rb") as f:
        # header token
        tok = '#!AER-DAT'
        # default version
        version = 0

        end_of_header = False
        bod = 0

        # read header and find file version number
        while not end_of_header:
            bod = f.tell()
            line = f.readline()

            if not line.startswith(b"#"):
                end_of_header = True
            else:
                if line.decode('utf-8').startswith((tok)):
                    version = int(float(line.decode('utf-8').strip().split(tok)[-1]))
                if debug:
                    print(line.decode('utf-8').strip())
        f.seek(bod)

        # set number of bytes per event based on the file version
        num_bytes_per_event = 6        

        # check version
        if version == 0:
            if debug:
                print('No #!AER-DAT version header found, assuming 16 bit addresses')
                version = 1
        elif version == 1:
            if debug:
                print('Addresses are 16 bit')
        elif version == 2:
            num_bytes_per_event = 8
            if debug:
                print('Addresses are 32 bit')
        else:
            if debug:
                print(f'Unknown file version {version}')            
        
        # seek to end of file
        f.seek(0, 2)
        # find the total number of bytes and number of events
        num_bytes = f.tell()
        num_events = (num_bytes - bod) // num_bytes_per_event
        offset = 0

        # seek to beginning of data
        f.seek(bod)        

        if num_events > max_events:
            if debug:
                print(f'clipping to {max_events} events although there are {num_events} events in file')
            # num_events = max_events
            offset = (num_events - max_events)*num_bytes_per_event

            # seek to new offset position to read the latest Max_events (read newer not older events)
            f.seek(offset)
            
        elif debug:
            print("Total number of events: {}".format(num_events))

        all_addr = None
        all_ts = None

        if version == 1: # needs to be verified --
            # read file (from bod) in big-endean unsigned 16-bit integer format 
            file_data = np.fromfile(f, dtype=">u2")
            # extract address (x, y, p) data (read 2 bytes or 1 uint16 starting from byte 0 then skip 4 bytes or 2 uint16)
            all_addr = file_data[::3]

            # f.seek(bod + 2, os.SEEK_SET)
            # extract timestamp data (every other 4 bytes or 1 uint32 starting from byte 2 or uint16 1)
            all_ts = []
            i = 1
            while i < num_events:
                all_ts.append(file_data[i])
                all_ts.append(file_data[i+1])
                i += 3

        elif version == 2:
            # read file (from bod) in big-endean unsigned 32-bit integer format 
            file_data = np.fromfile(f, dtype=">u4")
            # extract address (x, y, p) data (every other 4 bytes or 1 uint32 starting from byte 0)
            all_addr = file_data[::2]
            # extract timestamp data (every other 4 bytes or 1 uint32 starting from byte 4)
            all_ts = file_data[1::2]

        else:
            raise(ValueError(f'Unknown file version {version}'))

    # some verifications
    if all_addr is None or all_ts is None:
        raise(ValueError('all_addr or all_ts is None.'))

    if len(all_addr) != len(all_ts):
        raise(ValueError('length of all_addr does not equal length of all_ts.'))

    if num_events == 0:
        raise(ValueError('Number of events found is equal to zero.'))

    # retina_size_x=128

    # Initialize dictionary to store event data with static array sizes (vital for optimal run-time performance)
    td_data = {'ts': all_ts,\
                    'x': np.ndarray(num_events, dtype=np.uint16), \
                    'y': np.ndarray(num_events, dtype=np.uint16), \
                    'p': np.ndarray(num_events, dtype=np.int8)}

    xmask = int('fE', 16)  # x are 7 bits (64 cols) ranging from bit 1-7
    ymask = int('7f00', 16)  # y are also 7 bits ranging from bit 8 to 14.
    xshift = 1  # bits to shift x to right
    yshift = 8  # bits to shift y to right
    # polmask = 1  # polarity bit is LSB   

    # rotate frame clockwise by 90 degrees (flip x and y coordinates, and subtract new x from original frame height)
    if rotate90:
        # extract each event's pixel x address
        # td_data["x"][:] = retina_size_x - 1 - ((all_addr.astype(np.uint16) & ymask) >> yshift)
        td_data["x"][:] = 127 - ((all_addr.astype(np.uint16) & ymask) >> yshift)
        # extract each event's pixel y address
        # td_data["y"][:] = retina_size_x - 1 - ((all_addr.astype(np.uint16) & xmask) >> xshift)
        td_data["y"][:] = 127 - ((all_addr.astype(np.uint16) & xmask) >> xshift)
    else:
        # extract each event's pixel x address
        # td_data["x"][:]  = retina_size_x -1 - ((all_addr.astype(np.uint16) & xmask) >> xshift)
        td_data["x"][:]  = 127 - ((all_addr.astype(np.uint16) & xmask) >> xshift)
        # extract each event's pixel y address
        td_data["y"][:] = (all_addr.astype(np.uint16) & ymask) >> yshift
    # find polarity & convert to desired format {"default": -1, +1, "binary": 0, +1, "one-two": +1, +2}
    p_data = (all_addr.astype(np.int8) & 1)^1
    td_data["p"] = p_data if polarity_mode == "binary" else (p_data * 2) - 1 if polarity_mode == "default" else p_data + 1 if polarity_mode == "one-two" else p_data
    
    # deprecated slower methods
    # p_data = -1 * np.bitwise_and(all_addr, polmask).astype('int16') + 1
    # p_data = np.sign(np.bitwise_and(all_addr, polmask).astype('int16')) * -1
    # p_data = (all_addr & polmask).astype('int16')* -1 + 1

    if debug:
        end_t = time.time()
        print('time to process file =', end_t-start_t, 'seconds')
    return td_data