import struct 

# helper function to write events (in a dictionary format) to aedat file compatible with read_aedat function
def write_events_to_file(out_file, events_dict):
    # find number of events in the provided dict
    num_events = len(events_dict['x'])    
    
    # Open output file and write header
    with open(out_file, 'wb') as f:
        # Write header
        f.write(struct.pack('H', 0))  # eventType
        f.write(struct.pack('H', 0))  # eventSource
        f.write(struct.pack('I', 28))  # eventSize
        f.write(struct.pack('I', 0))  # eventTSOffset
        f.write(struct.pack('I', 0))  # eventTSOverflow
        f.write(struct.pack('I', num_events))  # eventCapacity
        f.write(struct.pack('I', num_events))  # eventNumber
        f.write(struct.pack('I', 1))  # eventValid

        # Loop over event data and write events to output file
        for i in range(num_events):
            ts = events_dict['ts'][i]
            x = events_dict['x'][i]
            y = events_dict['y'][i]
            p = events_dict['p'][i]

            # If the event falls within the label start and end times, write it to the output file
            data = (x << 17) | (y << 2) | (p << 1)
            f.write(struct.pack('II', data, ts))