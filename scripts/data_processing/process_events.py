import bisect
from config.representations_config import representations

def generate_event_representation(events_dict, width, height, delta_t=0, representation ='cstr', channels = 3, polarity_mode = 'binary'):
    """
    Generates a fixed-size event representation based on the specified method from a batch of events.

    This function dynamically selects an event representation method from a predefined list and applies it to the provided batch of event data. The event data is expected to be in a dictionary format containing timestamps, x and y coordinates, and polarity information.

    Parameters:
    - events_dict: A dictionary with lists of timestamps ('ts'), x coordinates ('x'), y coordinates ('y'), and polarities ('p').
    - width, height: Dimensions of the output representation.
    - delta_t: The duration of the event batch window in milliseconds. If 0, the full duration from the first to the last event is used.
    - representation: The method used for event representation. Defaults to 'cstr'.
    - channels: The number of channels in the output representation. Defaults to 3.
    - polarity_mode: Specifies how polarities are treated. Defaults to 'binary'.

    Returns:
    - A numpy array representing the processed event data according to the selected method.

    Raises:
    - ValueError: If the specified representation method is not supported.
    """    
    assert width > 0 and height > 0, "Width and height must be positive integers."

    # Get the timestamp of the last event in the event batch
    last_event_ts = events_dict['ts'][-1]

    # Determine the event window start time for event processing (set to 0 if delta_t was not specified)
    window_start_time = 0 if delta_t == 0 else last_event_ts - delta_t

    # Filter old events if a specific time window (event sampling duration) is set
    if delta_t > 0:
        filter_old_events(events_dict, window_start_time)
    
    # Otherwise measure the batch's sample period delta_t for the given event batch
    else:
        delta_t = last_event_ts - window_start_time

    # Call the specified event representation function
    if representation in representations:
        return representations[representation](events_dict, delta_t, height, width, channels, polarity_mode, window_start_time=window_start_time)
    else:
        raise ValueError(f"Event Representation [{representation}] not supported!")

# helper function to filter out old events based on specified start time
def filter_old_events(events_dict, window_start_time):
    """
    Filters out events from the event batch (in dictionary format) that are older than the specified start time.

    Parameters:
    - events_dict: The dictionary of event data containing 'ts' (timestamps), 'x' and 'y' (coordinates), and 'p' (polarities).
    - window_start_time: The start time for the event window, used to filter out older events.

    The function directly modifies the input dictionary, removing entries that precede the window_start_time.
    """    
    
    # Find the index of the first event within the specified time window using bisection method
    index = bisect.bisect_left(events_dict['ts'], window_start_time)

    # Remove events before the specified start time based on first event's index
    for key in ['ts', 'x', 'y', 'p']:
        events_dict[key] = events_dict[key][index:]

    assert len(events_dict['ts']) == len(events_dict['x']) == len(events_dict['y'])== len(events_dict['p'])