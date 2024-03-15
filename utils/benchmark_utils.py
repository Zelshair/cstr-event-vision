from config.config import Configs

# Helper function to remove inceptionV3 entries from list of the classifiers and provide them in a separate list
def list_inception_classifiers(classifiers):
    inception_classifiers = []

    inceptionv3, inceptionv3_aux = ('inceptionv3', 'inceptionv3_aux')

    if inceptionv3 in classifiers:
        classifiers.remove(inceptionv3)
        inception_classifiers.append(inceptionv3)

    if inceptionv3_aux in classifiers:
        classifiers.remove(inceptionv3_aux)
        inception_classifiers.append(inceptionv3_aux)
    
    return inception_classifiers

def create_config(dataset, event_rep, channels, keep_size, **kwargs):
    # Define a list of valid parameter names expected by Configs
    valid_params = ['dataset', 'event_rep', 'channels', 'keep_size', 'save_results', 'cache_dataset', 
                    'cache_transforms', 'cache_test_set', 'normalization', 'balanced_splits', 
                    'delta_t', 'augmentation_config', 'classifier', 'pretrained_classifier']

    # Filter kwargs to only include valid Configs() object parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return Configs(dataset=dataset, event_rep=event_rep, channels=channels, keep_size=keep_size, **filtered_kwargs)