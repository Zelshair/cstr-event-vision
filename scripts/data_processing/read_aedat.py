"""
Note: This assumes that timestamps have no offset.
Timestamps are read using an unsigned 32-bit integers cover up to 4295 seconds (1:11 Hours) if offset is removed.
Otherwise, might overflow and cause wrong results.

"""
import struct
import numpy as np

def read_aedat_file(filename, debug=False):
    # Open file in binary mode
    with open(filename, 'rb') as f:
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

        # find the total number of events
        total_events = 0
        while True:
            # Read header
            header = f.read(28)
            if not header:
                break  # End of file

            # Unpack header data
            _, _, _, _, _, _, eventNumber, _ = struct.unpack('HHIIIIII', header)

            # add number of events available in the current packet
            total_events += eventNumber

            # skip the event data
            f.seek(f.tell() + eventNumber*2*4)

        if debug:
            print("Total events =", total_events)

        # Initialize dictionary to store event data with static array sizes (vital for optimal run-time performance)
        events_dict = {'ts': np.ndarray(total_events, dtype=np.uint32),\
                        'x': np.ndarray(total_events, dtype=np.uint16), \
                        'y': np.ndarray(total_events, dtype=np.uint16), \
                        'p': np.ndarray(total_events, dtype=np.int8)}

        # return to start of data
        f.seek(bod)

        # set index
        i = 0

        # Read events and add to the dictionary of events
        while i <= total_events:
            # Read header
            header = f.read(28)
            if not header:
                break  # End of file

            # Unpack header data
            _, _, _, _, _, _, eventNumber, _ = struct.unpack('HHIIIIII', header)

            # Read events (2 uint32/4 bytes per each event)
            event_data = np.fromfile(f, dtype=np.uint32, count=eventNumber*2)
            if not event_data.size:
                break  # End of file

            # Extract the event data
            # ts: every other uint32 is ts data
            ts = event_data[1::2]
            # x,y,p data are provided in every other uint32
            data = event_data[::2]

            # Extract x, y, and polarity from data
            x = ((data >> 17) & 0x00001FFF).astype(np.uint16)
            y = ((data >> 2) & 0x00001FFF).astype(np.uint16)
            p = ((data >> 1) & 0x00000001).astype(np.int8)            

            # Append data to dictionary
            events_dict['ts'][i:i+eventNumber] = ts
            events_dict['x'][i:i+eventNumber]  = x
            events_dict['y'][i:i+eventNumber]  = y
            events_dict['p'][i:i+eventNumber]  = p

            # update index
            i += eventNumber

    return events_dict