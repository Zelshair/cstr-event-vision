'''
Python version of the matlab function load_atis_data.m for reading Prophesee's N-Cars dataset, developed by Zaid El Shair.

Event data files are binary files (.data) with the following format:
Each example is a separate binary file consisting of a list of events. First 3 lines contain the header. Afterwards 4 bytes of time data, 4 bytes of address data (x,y,p):

'''

import numpy as np

def load_atis_data(filename, flipX=0, flipY=0, polarity_mode="binary"):
#     """
#     This function reads a .dat file of event data in the N-Cars format as input and returns a dict containing event data in desired format
    
#     Parameters:
#         filename: The .dat binary filename containing the event data for N-Cars dataset
    
#     Returns:
#         dict: dictionary containing event data. Keys = "x", "y", "p", "ts"

#     description from the original matlab .m script:
#     % Loads data from files generated by the ATIS CAMERA
#     % This function only read (t,x,y,p) and discard other fields (if
#     % any) of events.
#     % timestamps are in uS
#     % td_data is a structure containing the fields ts, x, y and p
#     %
#     % flipX, flipY allow to flip the image arround the X and Y axes. If these values
#     are non zero, the corresponding dimension will be flipped considering its size
#     to be the value contained in the 'flip' variable (i.e. X = flipX - X)
#     (They defaults to 0 if non-specified)
#     """    

    with open(filename, "rb") as f:
        # Parse header if any
        header = []
        end_of_header = False
        num_comment_line = 0
        while not end_of_header:
            bod = f.tell()
            tline = f.readline()
            if not tline.startswith(b"%"):
                end_of_header = True
            else:
                words = tline.split()
                if len(words) > 2:
                    if words[1] == b"Date":
                        if len(words) > 3:
                            header.append((words[1], b" ".join(words[2:])))
                    else:
                        header.append((words[1], words[2]))
                num_comment_line += 1
        f.seek(bod)

        ev_type = 0
        ev_size = 8
        if num_comment_line > 0: # ensure compatibility with previous files.
            # Read event type
            ev_type = ord(f.read(1))
            # Read event size
            ev_size = ord(f.read(1))

        bof = f.tell()
        
        f.seek(0, 2)
        num_events = (f.tell() - bof) // ev_size

        # read data
        f.seek(bof)

        # read all event data in bytes from the 3rd line (after the header) in little-endian format,
        # "<" represents "little-endian" and "u4" represents "unsigned integer of 4 bytes"
        file_data = np.fromfile(f, dtype="<u4")

        # extract timestamp data (every other 4 bytes or 1 uint32 starting from byte 0)
        all_ts = file_data[::2]
        # extract address (x, y, p) data (every other 4 bytes or 1 uint32 starting from byte 4)        
        all_addr = file_data[1::2]

    # create output dict
    td_data = dict(ts=all_ts, x=[], y=[], p=[])

    version = 0
    for index, (key, value) in enumerate(header):
        if key == b"Version":
            version = value
            break

    xmask = int("00003FFF", 16)
    ymask = int("0FFFC000", 16)
    polmask = int("10000000", 16)
    xshift = 0  # bits to shift x to right
    yshift = 14  # bits to shift y to right
    polshift = 28  # bits to shift p to right

    addr = np.abs(all_addr)  # make sure nonnegative or an error will result from bitand (glitches can somehow result in negative addressses...)
    td_data["x"] = (addr & xmask) >> xshift
    td_data["y"] = (addr & ymask) >> yshift
    # find polarity & convert to desired format {"default": -1, +1, "binary": 0, +1, "one-two": +1, +2}
    p_data = ((addr & polmask) >> polshift)
    td_data["p"] = (p_data if polarity_mode == "binary" else (p_data * 2) - 1 if polarity_mode == "default" else p_data + 1 if polarity_mode == "one-two" else p_data).astype('int32')
    
    # original find polarity line (only default mode)
    # td_data["p"] = -1 + 2 * ((addr & polmask) >> polshift)

    if flipX > 0:
        td_data["x"] = flipX - td_data["x"]

    if flipY > 0:
        td_data["y"] = flipY - td_data["y"]

    return td_data