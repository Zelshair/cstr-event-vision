"""
Configuration for event representation methods.

This module defines a dictionary mapping the names of event representation methods
to their corresponding function implementations. These representations are used
to process and transform event-based vision data into various formats for further analysis
or model training.

To add a new representation, simply add a dictionary entry in 'representations' containing
the representation generation function.
"""

from scripts.data_processing.representations import *

# Define Event Representation Functions here
representations = {
    'cstr' : cstr,
    'cstr-mean' : cstr_mean_timestamps,
    'bin_event_frame' : bin_event_frame,
    'pol_event_frame' : pol_event_frame,
    'bin_event_count' : bin_event_count,
    'pol_event_count' : pol_event_count,
    'timestamp_img' : timestamp_img,
    'timestamp_img-count' : timestamp_img_count,
    'median' : median,
    'medIQRcount' : medIQRcount,
    'medcount' : medcount,
    'meanSTDcount' : meanSTDcount,

    # Add other event representations function definitions here as needed
}