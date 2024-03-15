

class AugmentationConfig():
    """
    A configuration class for asynchronous event-based data augmentations.
    It stores flags for enabling/disabling spatial, temporal, and polarity type augmentations.

    Attributes:
        spatial (bool): Whether to enable spatial augmentations.
        temporal (bool): Whether to enable temporal augmentations.
        polarity (bool): Whether to enable polarity augmentations.
    """
    def __init__(self, spatial=False, temporal=False, polarity=False)-> None:
        """
        Initializes an instance of the AugmentationConfig class with the specified augmentation flags.

        Args:
            spatial (bool): Whether to enable spatial augmentations.
            temporal (bool): Whether to enable temporal augmentations.
            polarity (bool): Whether to enable polarity augmentations.
        """        
        self.spatial = spatial
        self.temporal = temporal
        self.polarity = polarity

        # set threshold for each augmentation type
        self.s_threshold = 0.5
        self.ts_threshold = 0.5
        self.p_threshold = 0.5

    # function to check if any of the augmentation types are enabled
    def is_enabled(self):
        """
        Returns True if any of the augmentation flags are enabled, False otherwise.

        Returns:
            bool: Whether any augmentation is enabled.
        """        
        return self.spatial or self.temporal or self.polarity
    
        # Helper function to print augmentation configuration
    def print_config(self):
        print('\n\tAugmentation Configuration:')
        print('\t\tEnabled Augmentations:', ', '.join([aug for aug, enabled in [('Spatial', self.spatial), ('Temporal', self.temporal), ('Polarity', self.polarity)] if enabled]))
        
        if self.spatial:
            print('\t\tSpatial Augmentation Parameters:')
            print(f'\t\t\t- probability: {round(self.s_threshold*100)}%')

        if self.temporal:
            print('\t\tTemporal Augmentation Parameters:')
            print(f'\t\t\t- probability: {round(self.ts_threshold*100)}%')
            # print(f'\t\t\t- Temporal-Drop Range: [0, {self.max_temp_drop*100}%]')

        if self.polarity:
            print('\t\tPolarity-Inversion Augmentation Parameters:')
            print(f'\t\t\t- probability: {round(self.p_threshold*100)}%')