import numpy as np
import math

def cstr(events_dict, delta_t, height, width, channels = 3, polarity_mode = 'binary', window_start_time = 0):
    """
    Generates a Compact Spatio-Temporal Representation (CSTR) from event data.

    This function creates a 3-channel representation of event data, where:
    - The first channel (red) encodes the average timestamp of negative events,
    - The second channel (green) encodes the count of events, normalized across the frame,
    - The third channel (blue) encodes the average timestamp of positive events.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Expected to be 3, following the RGB channel convention.
    - polarity_mode: Specifies how polarities are treated ('binary' expected).
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, 3) representing the encoded event data.
    """

    assert channels == 3, "CSTR representation requires 3 channels."

    # Initialize arrays for timestamp and event count processing
    event_rep_t = np.zeros((height, width, 2), dtype=np.float32)  # 2 channels for average timestamp calculation per polarity
    event_rep_count = np.zeros((height, width, 2), dtype=np.float32)  # 2 event count channels per polarity

    # Initialize the final 3-channel event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]

        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time)/delta_t  

        # Add normalized event time and increment event count at position x,y
        if polarity_mode == 'binary':
            event_rep_t[y][x][p] += t
            event_rep_count[y][x][p] += 1
        else:
            raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")
    
    # Calculate the average timestamp for indexes with non-zero counts
    nonzero_mask = event_rep_count != 0
    np.divide(event_rep_t, event_rep_count, out=event_rep_t, where=nonzero_mask)

    # Assign channels in the CSTR representation
    event_rep[..., 0] = event_rep_t[..., 0] # Average timestamp of negative events
    event_rep[..., 2] = event_rep_t[..., 1] # Average timestamp of positive events
    event_rep[..., 1] = np.sum(event_rep_count, axis=-1)  # Total event count

    # Normalize the event count channel
    event_rep[..., 1] /= event_rep[..., 1].max()
    
    return event_rep

def cstr_mean_timestamps(events_dict, delta_t, height, width, channels=2, polarity_mode='binary', window_start_time=0):
    """
    Generates a Compact Spatio-Temporal Representation (CSTR) using mean timestamps only, without incorporating event counts into the representation channels.

    This function creates a representation of event data focusing on the mean timestamps:
    - For a 1-channel output, the resulting single-channel output represents the mean timestamp of the events disregarding polarity.
    - For a 2-channel output, each channel represents the mean timestamp of negative and positive events, respectively.
    - For a 3-channel output (RGB), the red and blue channels represent the mean timestamps of negative and positive events, respectively. 
      The green channel is not used in this context.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Number of channels in the output representation, can be 1, 2, or 3.
    - polarity_mode: Specifies how polarities are treated ('binary' expected).
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, channels) representing the encoded event data.
    """

    assert channels in [1, 2, 3], "CSTR mean timestamps representation requires 1, 2, or 3 channels."

    # Initialize arrays for timestamp and event count processing
    event_rep_t = np.zeros((height, width, 2), dtype=np.float32)
    event_rep_count = np.zeros((height, width, 2), dtype=np.float32)    

    # Initialize the final event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]

        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time) / delta_t

        # Add normalized event time and increment event count at position x,y
        if polarity_mode == 'binary':
            event_rep_t[y][x][p] += t
            event_rep_count[y][x][p] += 1
        else:
            raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")

    # Assign channels in the CSTR mean-timestamps only representation based on the specified number of output channels
    if channels in [2,3]:
        # Calculate the average timestamp for indexes with non-zero counts for each polarity
        nonzero_mask = event_rep_count != 0
        np.divide(event_rep_t, event_rep_count, out=event_rep_t, where=nonzero_mask)        
        if channels == 2:
            # For 2 channels, directly use the mean timestamps
            event_rep[..., :2] = event_rep_t

        elif channels == 3:
            # If 3 channels are requested, the third channel remains unused/zero
            event_rep[..., 0] = event_rep_t[..., 0] # Average timestamp of negative events
            event_rep[..., 2] = event_rep_t[..., 1] # Average timestamp of positive events

    elif channels == 1:
        # For 1 channel, find the sum of the timestamps and the event count for all polarities
        bin_event_rep_t = np.sum(event_rep_t, axis = -1)
        bin_event_rep_count = event_rep_count[..., 0] + event_rep_count[..., 1]

        # Calculate the average timestamp for indexes with non-zero counts for each polarity
        nonzero_mask = bin_event_rep_count != 0
        event_rep[..., 0] = np.divide(bin_event_rep_t, bin_event_rep_count, where=nonzero_mask)

    return event_rep

def timestamp_img(events_dict, delta_t, height, width, channels=2, polarity_mode='binary', window_start_time=0):
    """
    Generates an image representation where each pixel encodes the most recent event's normalized timestamp at that location.

    The output can be configured for different numbers of channels:
    - For a 1-channel output, it represents the most recent event timestamp regardless of polarity
    - For a 2-channel output, each channel represents the most recent timestamp of negative and positive events, respectively.
    - For a 3-channel output (RGB), the red and blue channels represent the most recent timestamps of negative and positive events, respectively, with the green channel unused.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Number of channels in the output representation, can be 1, 2, or 3.
    - polarity_mode: Specifies how polarities are treated ('binary' expected).
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, channels) representing the encoded event data.
    """
    
    assert channels in [1, 2, 3], "timestamp_img representation requires 1, 2, or 3 channels."

    # Initialize arrays for storing the most recent timestamp per polarity
    event_rep_t = np.zeros((height, width, 2), dtype=np.float32)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]
        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time) / delta_t

        # Update the pixel with the most recent event timestamp
        if polarity_mode == 'binary':
            event_rep_t[y, x, p] = t  # Always keep the latest timestamp
        else:
            raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")

    # Initialize the final event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    if channels == 2:
        # For 2 channels, directly use the most recent timestamps for each polarity
        event_rep = event_rep_t

    elif channels == 3:
        # For 3 channels, use the most recent timestamps for negative and positive events in the red and blue channels, respectively
        event_rep[..., 0] = event_rep_t[..., 0]  # Most recent timestamp of negative events
        event_rep[..., 2] = event_rep_t[..., 1]  # Most recent timestamp of positive events
        # The green channel remains unused

    elif channels == 1:
        # For 1 channel, find the max of the most recent timestamps across polarities
        event_rep[..., 0] = np.max(event_rep_t, axis=-1)

    return event_rep

def timestamp_img_count(events_dict, delta_t, height, width, channels=2, polarity_mode='binary', window_start_time=0):
    """
    Generates an image representation where each pixel encodes the most recent event's normalized timestamp at that location per polarity in addition to the event count.

    This function creates a 3-channel representation of event data, where:
    - The first channel (red) encodes the most recent timestamp of negative events,
    - The second channel (green) encodes the count of events, normalized across the frame,
    - The third channel (blue) encodes the most recent timestamp of positive events.
    
    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Expected to be 3, following the RGB channel convention.
    - polarity_mode: Specifies how polarities are treated ('binary' expected).
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, channels) representing the encoded event data.
    """
    
    assert channels == 3, "Timestamp Image & Count representation requires 3 channels."

    # Initialize arrays for storing the most recent timestamp per polarity and event count processing
    event_rep_t = np.zeros((height, width, 2), dtype=np.float32)  # 2 channels for most recent timestamp per polarity
    event_rep_count = np.zeros((height, width, 2), dtype=np.float32)  # 2 event count channels per polarity

    # Initialize the final event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]

        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time) / delta_t

        # Update the pixel with the most recent event timestamp and increment event count based on its polarity
        if polarity_mode == 'binary':
            event_rep_t[y][x][p] = t  # Always keep the latest timestamp
            event_rep_count[y][x][p] += 1
        else:
            raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")

    # Assign channels in the Timestamp_image & Count representation
    event_rep[..., 0] = event_rep_t[..., 0]  # Most recent timestamp of negative events
    event_rep[..., 2] = event_rep_t[..., 1]  # Most recent timestamp of positive events
    event_rep[..., 1] = np.sum(event_rep_count, axis=-1)  # Total event count

    # Normalize the event count channel
    event_rep[..., 1] /= event_rep[..., 1].max()
    
    return event_rep

def bin_event_frame(events_dict, delta_t, height, width, channels = 1, polarity_mode = 'binary', window_start_time = 0):
    return event_frame(events_dict, height, width, channels, polarity=False, polarity_mode=polarity_mode)

def pol_event_frame(events_dict, delta_t, height, width, channels = 2, polarity_mode = 'binary', window_start_time = 0):
    return event_frame(events_dict, height, width, channels, polarity=True, polarity_mode=polarity_mode)

def event_frame(events_dict, height, width, channels = 1, polarity = False, polarity_mode = 'binary'):
    """
    Generates an event frame representation indicating event occurrences at each pixel location.
    This function creates either a binary (non-polarized) or polarized event frame representation, based on the 'polarity' parameter:
    - A Binary Event Frame: supports 1, 2, or 3 channel outputs, indicating the presence of events without considering polarity.
    - A Polarized Event Frame: supports 1, 2, or 3 channel outputs, indicating the presence of events based on polarity.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Number of channels in the output representation, can be 1, 2, or 3.
    - polarity: If True, generates a Polarized Event Frame representation. If False, polarity is ignored and generates Binary Event Frame representation instead.
    - polarity_mode: Specifies how polarities are treated ('binary' expected).
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, channels) representing the encoded event data.
    """
    assert channels in [1, 2, 3], "Event Frame representation requires 1, 2, or 3 channels."

    # Initialize the event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]

        # Generate Polarized Event Frame
        if polarity:
            if polarity_mode == 'binary':
                if channels == 1 or channels == 3:
                    if p == 1:
                        event_rep[y][x][0:channels] = 1
                    elif p == 0:
                        event_rep[y][x][0:channels] = -1                      

                elif channels == 2:
                    event_rep[y][x][p] = 1

            else:
                raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")
        
        # Generate Binary Event Frame
        else:
            # Set pixel coordinate (x,y) as 1 (indicating that an event has occured at this pixel)
            event_rep[y][x][0:channels] = 1

    return event_rep

def bin_event_count(events_dict, delta_t, height, width, channels = 1, polarity_mode = 'binary', window_start_time = 0):
    return event_count(events_dict, height, width, channels, polarity=False, polarity_mode=polarity_mode)

def pol_event_count(events_dict, delta_t, height, width, channels = 2, polarity_mode = 'binary', window_start_time = 0):
    return event_count(events_dict, height, width, channels, polarity=True, polarity_mode=polarity_mode)

def event_count(events_dict, height, width, channels = 1, polarity = False, polarity_mode = 'binary'):
    """
    Generates an Event Count representation indicating the number of event occurrences at each pixel position.
    This function creates either a binary (non-polarized) or polarized event count representation, based on the 'polarity' parameter:
    - A Binary Event Count: supports 1, 2, or 3 channel outputs, indicating the number of events regardless of polarity.
    - A Polarized Event Count: supports 2 or 3 channel outputs only, differentiating between the number of positive and negative events.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Number of channels in the output representation, can be 1, 2, or 3.
    - polarity: If True, generates a Polarized Event Count representation. If False, polarity is ignored and generates Binary Event Count representation instead.
    - polarity_mode: Specifies how polarities are treated ('binary' expected).
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, channels) representing the encoded event data.
    """
    
    # Validation of channel support based on polarity
    if polarity:
        assert channels in [2, 3], "Polarized Event Count representation suports 2 or 3 channel outputs only."
    else:
        assert channels in [1, 2, 3], "Binary Event Count representation suuports 1, 2, or 3 channel outputs only."

    # Initialize the event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]

        # Generate Polarized Event Count
        if polarity:
            # Increase the event count at pixel coordinate (x,y) by 1 based on polarity
            if polarity_mode == 'binary':
                if channels == 2:
                    event_rep[y][x][p] += 1
                elif channels == 3:
                    event_rep[y][x][p*2] += 1 # populate red and blue channels only
            else:
                raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")
        
        # Generate Binary Event Count
        else:
            # Increase the event count at pixel coordinate (x,y) by 1 
            event_rep[y][x][0:channels] += 1

    return event_rep

def meanSTDcount(events_dict, delta_t, height, width, channels = 3, polarity_mode = 'binary', window_start_time = 0):
    """
    Generates a Mean, STD Dev, and Count representation from event data.

    This function creates a 3-channel representation of event data, where:
    - The first channel (red) encodes the average timestamp of events,
    - The second channel (green) encodes the count of events normalized across the frame,
    - The third channel (blue) encodes the standard deviation of events' timestamps.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Expected to be 3, following the RGB channel convention.
    - polarity_mode: Specifies how polarities are treated ('binary' expected). Not utilized in this function.
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, 3) representing the encoded event data.
    """    
    assert channels == 3, "Mean-STD-Count representation requires 3 channels."

    # Initialize the final 3-channel event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # create array to keep track sum of squares for each index
    sum_squares_array = np.zeros((height, width), dtype=np.float32)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y = events_dict["x"][i], events_dict["y"][i]
        
        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time)/delta_t  

        # Add normalized event time and increment event count at position x,y
        event_rep[y][x][0] += t
        event_rep[y][x][1] += 1

        # Calculate the square of the event's timestamp
        sum_squares_array[y][x] += t**2
        
        # Get the latest event count at x,y
        n = event_rep[y][x][1]

        # Calculate standard deviation of timestamps per index per event polarity if more than 1 event have occurred
        if n > 1:
            # Calculate the mean of timestamps
            mean_squared = (event_rep[y][x][0] / n)**2
            # Calculate the mean of the squared timestamp values
            mean_squared_vals = sum_squares_array[y][x]/n
            # Measure  the difference
            diff = mean_squared_vals - mean_squared

            # for overflow floating point precision error (results in very small negative numbers)
            if diff > 0:
                event_rep[y][x][2] = math.sqrt(diff)

    # Calculate the average timestamp for indexes with non-zero counts
    nonzero_mask = event_rep[...,1] != 0
    event_rep[..., 0] = np.divide(event_rep[..., 0], event_rep[..., 1], where=nonzero_mask)
    
    # Normalize the event count channel
    event_rep[..., 1] /= event_rep[..., 1].max()

    return event_rep

def median(events_dict, delta_t, height, width, channels = 2, polarity_mode = 'binary', window_start_time = 0):
    """
    Generates a Median representation from event data using median timestamps only.

    This function creates a representation of event data focusing on the median timestamps:
    - For a 1-channel output, the resulting single-channel output represents the median timestamp of the events disregarding polarity.
    - For a 2-channel output, each channel represents the median timestamp of negative and positive events, respectively.
    - For a 3-channel output (RGB), the red and blue channels represent the median timestamps of negative and positive events, respectively. 
      The green channel is not used in this context.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Number of channels in the output representation, can be 1, 2, or 3.
    - polarity_mode: Specifies how polarities are treated ('binary' expected). Not utilized in this function.
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, channels) representing the encoded event data.
    """

    assert channels in [1, 2, 3], "Median Timestamp representation requires 1, 2, or 3 channels."
    
    # Initialize the final event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Create a multi-dimensional array to hold all the events timestamps for each spatial index
    if channels != 1:
        timestamps = np.empty((height, width, 2), dtype=np.ndarray)
    else:
        timestamps = np.empty((height, width), dtype=np.ndarray)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]

        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time) / delta_t

        if channels != 1:
            # Append event's timestamp to spatial index (x,y) based on polarity
            if polarity_mode == 'binary':
                # Check if index was not populated before
                if timestamps[y, x, p] is None:
                    # Create list of event timestamps
                    timestamps[y, x, p] = [t]
                else:
                    # Append latest timestamp to list at x,y,p
                    timestamps[y, x, p] = np.append(timestamps[y, x, p], t)
            else:
                raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")
                    
        else:
            # Check if index was not populated before
            if timestamps[y, x] is None:
                # Create list of event timestamps
                timestamps[y, x] = [t]
            else:
                # Append latest timestamp to list at x,y
                timestamps[y, x] = np.append(timestamps[y, x], t)
    
    # Loop over all frame indexes to find median (assuming timestamps are sorted)
    for i in range(timestamps.shape[0]):
        for j in range(timestamps.shape[1]):
            if channels != 1:
                for p in range(timestamps.shape[2]):                
                    # Check if no events occurred at spatial index (x,y)
                    if timestamps[i, j, p] is None:
                        timestamps[i, j, p] = 0
                    else:
                        # Find the mid index of the array
                        mid_index = len(timestamps[i, j, p]) // 2
                        # Find median (works for even and odd length arrays)
                        timestamps[i, j, p] = (timestamps[i, j, p][mid_index] + timestamps[i, j, p][~mid_index]) / 2
            else:
                # check if no events occurred at spatial index (x,y)
                if timestamps[i, j] is None:
                    timestamps[i, j] = 0
                else:
                    mid_index = len(timestamps[i, j]) // 2
                    # find median (works for even and odd length arrays)
                    timestamps[i, j] = (timestamps[i, j][mid_index] + timestamps[i, j][~mid_index]) / 2                

    # default representation's number of channels
    if channels == 2:
        event_rep[..., 0] = timestamps[..., 0]
        event_rep[..., 1] = timestamps[..., 1]

    # 3-channel representation (assumes RGB image encoding)
    elif channels == 3:
        # set red channel (1st) to be equal to negative events' median timestamp
        event_rep[..., 0] = timestamps[..., 0]
        # set blue channel (3rd) to be equal to positive event' median timestamp
        event_rep[..., 2] = timestamps[..., 1]        
    
    # 1-channel representation (ignores polarity)
    elif channels == 1:
        event_rep[..., 0] = timestamps
    
    return event_rep

def medIQRcount(events_dict, delta_t, height, width, channels = 3, polarity_mode = 'binary', window_start_time = 0):
    """
    Generates an Median-IQR-Count representation where each pixel encodes the median event timestamp, IQR, and count at that location.

    This function creates a 3-channel representation of event data, where:
    - The first channel (red) encodes the median timestamp of events,
    - The second channel (green) encodes the count of events, normalized across the frame,
    - The third channel (blue) encodes the IQR of the events' timestamps.
    
    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Expected to be 3, following the RGB channel convention.
    - polarity_mode: Specifies how polarities are treated ('binary' expected). Not utilized in this function.
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, channels) representing the encoded event data.
    """
        
    """3-channel representations: finds median, IQR and event count. Ignores event polarity information"""

    assert channels == 3, "Median-IQR-Count representation requires 3 channels."
    
    # Initialize the final 3-channel event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Create a multi-dimensional array to hold all the events timestamps for each spatial index
    timestamps = np.empty((height, width), dtype=np.ndarray)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y = events_dict["x"][i], events_dict["y"][i]

        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time)/delta_t  

        # Increment event count at spatial index (x, y) using the green channel, i.e. channel 1
        event_rep[y][x][1] += 1

        # Append the event's normalized timestamp to the spatial index's list at (x,y) 
        if timestamps[y, x] is None:
            # Create list of event timestamps
            timestamps[y, x] = [t]
        else:
            # Append latest timestamp to list at x,y
            timestamps[y, x] = np.append(timestamps[y, x], t)

    # Normalize the event count channel
    event_rep[..., 1] /= event_rep[..., 1].max()
    
    # Loop over all frame indexes to find median and IQR(assuming timestamps are sorted)
    for i in range(timestamps.shape[0]):
        for j in range(timestamps.shape[1]):
            # check if no events occurred at spatial index (x,y)
            if timestamps[i, j] is None:
                # set median and IQR value to be equal to 0
                event_rep[i, j, 0] = 0
                event_rep[i, j, 2] = 0

            else:
                mid_index = len(timestamps[i, j]) // 2
                # find median (works for even and odd length arrays)
                event_rep[i, j, 0] = (timestamps[i, j][mid_index] + timestamps[i, j][~mid_index]) / 2         

                # set IQR channel at (i,j)
                if len(timestamps[i, j]) < 4:
                    event_rep[i, j, 2] = 0
                else:
                    q1, q3 = np.percentile(timestamps[i, j], [25, 75])

                    # calculate IQR
                    event_rep[i, j, 2] = q3 - q1                    

    return event_rep

def medcount(events_dict, delta_t, height, width, channels = 2, polarity_mode = 'binary', window_start_time = 0):
    """
    Generates a Median-Count Representation from event data.

    This function creates a 3-channel representation of event data, where:
    - The first channel (red) encodes the median timestamp of negative events,
    - The second channel (green) encodes the count of events, normalized across the frame,
    - The third channel (blue) encodes the median timestamp of positive events.

    Parameters:
    - events_dict: A dictionary containing lists of timestamps ('ts'), x and y coordinates ('x', 'y'), and polarities ('p').
    - delta_t: The time sampling duration used for the given event batch in ms.
    - height, width: The dimensions of the output representation.
    - channels: Expected to be 3, following the RGB channel convention.
    - polarity_mode: Specifies how polarities are treated ('binary' expected).
    - window_start_time: The start time of the event batch window, used for removing the time offset.

    Returns:
    - A numpy array of shape (height, width, 3) representing the encoded event data.
    """

    assert channels == 3, "Median-Count representation requires 3 channels."
    
    # Initialize the final 3-channel event representation array
    event_rep = np.zeros((height, width, channels), dtype=np.float32)

    # Create a multi-dimensional array to hold all the events timestamps for each spatial index and polarity
    timestamps = np.empty((height, width, 2), dtype=np.ndarray)

    # Process each event in the dictionary
    for i in range(len(events_dict["ts"])):
        x, y, p = events_dict["x"][i], events_dict["y"][i], events_dict["p"][i]

        # Remove event's time offset then normalize by dividing it by the batch-sampling duration "delta_t"
        t = (events_dict["ts"][i] - window_start_time)/delta_t  

        # Increment event count at spatial index (x, y) using the green channel, i.e. channel 1
        event_rep[y][x][1] += 1        

        # append event's timestamp to spatial index (x,y) based on polarity
        if polarity_mode == 'binary':
            # Check if index was not populated before
            if timestamps[y, x, p] is None:
                # Create list of event timestamps
                timestamps[y, x, p] = [t]
            else:
                # Append latest timestamp to list at x,y,p
                timestamps[y, x, p] = np.append(timestamps[y, x, p], t)
    
    # Normalize event count channel
    event_rep[..., 1] /= event_rep[..., 1].max()

    if polarity_mode == 'binary':
        # Loop over all frame indexes to find median (assuming timestamps are sorted)
        for i in range(timestamps.shape[0]):
            for j in range(timestamps.shape[1]):
                for p in range(timestamps.shape[2]):
                    # Find if no events occurred
                    if timestamps[i, j, p] is None:
                        event_rep[i, j, p*2]  = 0
                    else:
                        # Find the mid index of the array
                        mid_index = len(timestamps[i, j, p]) // 2
                        # Find median (works for even and odd length arrays)
                        event_rep[i, j, p*2] = (timestamps[i, j, p][mid_index] + timestamps[i, j, p][~mid_index]) / 2
    else:
        raise NotImplementedError(f"Polarity mode [{polarity_mode}] has not been implemented.")
            
    return event_rep