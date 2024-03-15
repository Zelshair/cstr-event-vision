"""class object for normalization parameters per the specified dataset, event rep and number of channels.

from utils.get_norm_params, get_norm_params() function was used to generate the normalization parameters below for each configuration.
"""

class NormParams():
    ''' Object to return normalization parameters (mean, std) based on the specified configuration.
    
    Notes:

    - if dataset_name is set to None, returns mean and std of 0.

    - if dataset_name is set to 'ImageNet', returns ImageNet mean and std based on number of channels specified.

    - otherwise, returns normalization parameter based on the specified dataset, event represnetation, and number of channels.

    - Event based datasets supported: 'N-MNIST', 'N-Cars'.
    
    - Event representations supported: 'cstr', 'event_frame', 'event_count'.
    '''
    def __init__(self, dataset_name='ImageNet', event_rep = 'clstr', channels=3) -> None:
        if dataset_name is not None:
            assert dataset_name in ['ImageNet', 'N-MNIST', 'N-Cars', 'N-Caltech101', 'CIFAR10-DVS'], "Dataset \"{}\" is not supported!".format(dataset_name)
            # assert dataset_name in dataset_config.supported_datasets, "Dataset \"{}\" is not supported!".format(dataset_name)
            assert event_rep in ['clstr', 'cstr', 'event_frame', 'event_count'], "Event Representation \"{}\" is not supported!".format(event_rep)
        # set dataset name
        self.dataset_name = dataset_name

        self.event_rep = event_rep
        self.channels = channels

        # intialize mean and std
        self.mean = [0.0 for _ in range(self.channels)] 
        self.std = [1.0 for _ in range(self.channels)] 

        # if default ImageNet normalization parameters are selected
        if self.dataset_name == 'ImageNet':
            if self.channels == 1:
                self.mean = [(0.485 + 0.456 + 0.406)/3]
                self.std = [(0.229 +  0.224 + 0.225)/3]

            elif self.channels == 2:
                self.mean = [(0.485 + 0.456 + 0.406)/3, (0.485 + 0.456 + 0.406)/3]
                self.std = [(0.229 +  0.224 + 0.225)/3, (0.229 +  0.224 + 0.225)/3]

            elif self.channels == 3:
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]

            else:
                raise ValueError("Number of channels = {} is not supported! Please select one of the following: 1, 2, 3".format(self.channels))


        # Otherwise, load specific configuration's normalization parameters
        elif self.dataset_name == 'N-MNIST':
            if self.event_rep == 'clstr':
                if self.channels == 1:
                    self.mean = [0.11866538]
                    self.std = [0.4464249]

                elif self.channels == 2:
                    self.mean = [0.19133517, 0.22336191]
                    self.std = [0.32619002, 0.3372072]

                elif self.channels == 3:                                
                    self.mean = [0.11866538, 0.11866538, 0.11866538]
                    self.std = [0.4464249, 0.4464249, 0.4464249]
                
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 1, 2, 3".format(self.channels))
            
            elif self.event_rep == 'cstr':
                if self.channels == 3:                                
                    self.mean = [0.14443131,0.1279256, 0.17392892]
                    self.std = [0.25489616, 0.21890034, 0.2688196]
                
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 3.".format(self.channels))                

        elif self.dataset_name == 'N-Cars':
            if self.event_rep == 'clstr':
                if self.channels == 1:
                    self.mean = [0.00351658]
                    self.std = [0.18333668]

                elif self.channels == 2:
                    self.mean = [0.02992869, 0.0370045]
                    self.std = [0.1271008, 0.14199041]

                elif self.channels == 3:                                
                    self.mean = [0.00351658, 0.00351658, 0.00351658]
                    self.std = [0.18333668, 0.18333668, 0.18333668]
                
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 1, 2, 3".format(self.channels))

            elif self.event_rep == 'cstr-mean':
                if self.channels == 1:
                    self.mean = [0.0]
                    self.std = [1.0]

                elif self.channels == 2:
                    self.mean = [0.02556231, 0.03217835]
                    self.std = [0.11126604, 0.12549697]  

                elif self.channels == 3:                                
                    self.mean = [0.0, 0.0, 0.0]
                    self.std = [1.0, 1.0, 1.0]
                
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 1, 2, 3".format(self.channels))

            elif self.event_rep == 'cstr':
                if self.channels == 3:                                
                    self.mean = [0.02556231, 0.0109354, 0.03217835]
                    self.std = [0.11126604, 0.04369345, 0.12549697]
                
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 3.".format(self.channels))        

        elif self.dataset_name == 'N-Caltech101':
            if self.event_rep == 'clstr':
                if self.channels == 1:
                    self.mean = [0.01190163]
                    self.std = [0.4734557]

                elif self.channels == 2:
                    self.mean = [0.22413558, 0.23165679]
                    self.std = [0.3199756, 0.3220562]

                elif self.channels == 3:                                
                    self.mean = [0.01190163, 0.01190163, 0.01190163]
                    self.std = [0.4734557, 0.4734557, 0.4734557]

            elif self.event_rep == 'cstr':
                if self.channels == 3:                   
                    self.mean = [0.17295973, 0.0065389,  0.17982708]
                    self.std = [0.25482053, 0.0133615,  0.25711596]
                
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 3.".format(self.channels))                            

        elif self.dataset_name == 'CIFAR10-DVS':
            if self.event_rep == 'clstr':
                if self.channels == 1:
                    self.mean = [-0.23565182]
                    self.std = [0.6545949]

                elif self.channels == 2:
                    self.mean = [0.5398909, 0.50185394]
                    self.std = [0.40944472, 0.40944472]

                elif self.channels == 3:                                
                    self.mean = [-0.23565182, -0.23565182, -0.23565182]
                    self.std = [0.6545949, 0.6545949, 0.6545949]
                          
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 1, 2, 3".format(self.channels))
                
            elif self.event_rep == 'cstr':
                if self.channels == 3:                   
                    self.mean = [0.30608723, 0.10910607,  0.2982166]
                    self.std = [0.23363586, 0.12710005,  0.24728328]
                
                else:
                    raise ValueError("Number of channels = {} is not supported! Please select one of the following: 3.".format(self.channels))        