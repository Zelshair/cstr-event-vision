# CSTR: A Compact Spatio-Temporal Representation for Event-Based Vision

This repository contains the source code for the paper "_**CSTR: A Compact Spatio-Temporal Representation for Event-Based Vision**_" published in **IEEE Access** ([link](https://ieeexplore.ieee.org/document/10254219)), developed by Zaid El Shair. Our work introduces a novel representation for event-based vision, significantly enhancing the processing and compatibility of event data with deep learning architectures.
![Graphical Abstract](https://github.com/Zelshair/cstr-event-vision/assets/50595418/3ec86fc6-84c2-4da0-8b0b-9c12852d4e27)
**Figure**: An event camera captures sparse and asynchronous events that represent brightness changes at each pixel in the frame. To process these events in batches and utilize modern computer vision architectures, an intermediate representation is required. In this work, we propose the compact spatio-temporal representation (CSTR) that encodes the event batch’s spatial, temporal, and polarity information in a 3-channel image-like format.

## Repository Structure
```
.
├── benchmark.py                       # Script for benchmarking different configurations
├── calculate_dataset_norms.py         # Calculates normalization parameters for datasets
├── calculate_dataset_statistics.py    # Computes statistical information of datasets
├── evaluate.py                        # Script for evaluating trained models
├── gen_dataset_csv.py                 # Generates CSV files for custom dataset support
├── split_dataset.py                   # Utility script for dataset splitting (if applicable)
├── split_dvs_gesture_seqs.py          # Splits DVS-Gesture sequences into sub-sequences
├── split_seqs_to_samples.py           # Splits sub-sequences into fixed-duration samples
├── train.py                           # Main script for training models
├── config                             # Configuration scripts and settings for the project
├── models                             # Definitions and utilities for deep learning models
├── output                             # Storage for training outputs like models and logs
├── scripts                            # Core scripts for data processing, training, and evaluation
│ ├── datasets                         # Dataset-specific processing and loading scripts
│ └── data_processing                  # Utilities for preprocessing and handling event data
├── utils                              # Supporting utilities for benchmarking, training, and evaluation
├── LICENSE                            # The license under which the project is released
└── README.md                          # Overview and documentation for this repository
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Other packages specified below.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Zelshair/cstr-event-vision.git
   ```

2. Create a new virtual environment and activate it (recommended using Conda)
   ```
   conda create -n cstr
   conda activate cstr
   ```
   
3. Install PyTorch (with GPU support if desired) by generating the right conda installation command using https://pytorch.org/get-started/locally/

4. Install dependencies:
   `pip install matplotlib scikit-learn pandas opencv-python numpy psutil torchmetrics pycocotools`

## Usage
### Training and Evaluating a Model

To train and evaluate a model using `train.py`, run the following command:
```
python train.py
```
This script will train a model with the configurations specified in config/config.py and then evaluate it using the test set defined in the same configuration file. The results, including accuracy and loss metrics, will be saved in the `output/results directory` (default location specified in `config.py`). Additionally, plots of the training history will be saved in the `output/training_history`.

### Evaluating a Trained Model
Otherwise, to evaluate a trained model without retraining, use the `evaluate.py` script. Ensure you specify the trained model filename for evaluation and setting the configuration in the configuration object `Configs()`, then run using the following command:
```
python evaluate.py
```

You can also use this function to visualize the testing results by setting `visualize_results` variable as `True` in the configuration object `Configs()`. Similarly, saving the evaluation results option can be set using the `save_results` variable.

### Training and Evaluating Multiple Models and Configurations
The `benchmark.py` script allows for training and evaluating multiple models and configurations efficiently by leveraging data caching. This is particularly useful for running extensive experiments:
```
python benchmark.py
```
Ensure to configure benchmark.py with the desired models, datasets, and training parameters using the function `benchmark_main()`. This script will train each model according to the specified configurations, evaluate them, and cache the data to accelerate the training process.

## Supported Representations
This project implements several image-like event representations, including **CSTR** as detailed in our publication, "_CSTR: A Compact Spatio-Temporal Representation for Event-Based Vision._" Currently supported representations include:

* **CSTR** (Compact Spatio-Temporal Representation): Reports the average timestamp of the events at each pixel, based on polarity, and the normalized number of events at each pixel.
* **CSTR mean timestamps only****: Reports the average timestamp of the events at each pixel.
* **Binary Event Frame**: Indicates the occurrence of an event at a location, disregarding its polarity.
* **Polarized Event Frame**: Similar to Binary Event Frame but based on the polarity of events.
* **Binary Event Count**: Counts the number of events occurring at each pixel location.
* **Polarized Event Count**: Similar to Binary Event Count but based on the polarity of events.
* **Timestamp Image**: Reports the most recent timestamp of events at each pixel, based on polarity, and the normalized number of events at each pixel.
* **Timestamp Image & Count**: Reports the most recent timestamp of events at each pixel, based on polarity, and the normalized number of events at each pixel.

The implementation of these representations can be found in `scripts/data_processing/representations.py`. Each representation's function requires that the events be provided in a `dict` format with a key per event component (`["x", "y", "p", "ts"]`). Additional representations such as Median, Median with Interquartile Range Count, and others are also implemented as detailed in `config/representations_config.py` and defined in `scripts/data_processing/representations.py`.

## Adding New Representations
To add a new event representation:
1. Implement Representation Function: Create a function in `scripts/data_processing/representations.py` that processes an event dictionary and generates the desired representation. Ensure your function accepts an `events_dict` parameter and any other relevant parameters for your representation.

2. Register Representation: Add your new representation method to config/representations_config.py by including an entry in the representations dictionary. Map your representation's name to the corresponding function you implemented in step 1.

Example for adding a new representation named "example_representation":

* In `representations.py`:
  
   ```
   def example_representation(events_dict, delta_t, height, width, channels):
       # Implementation of your representation
       return event_rep
   ```
* In `representations_config.py`:

   ```
   representations = {
       ...,
       'example_representation': example_representation,
       ...
   }
   ```

## Supported Datasets
This project supports various event-based datasets for both object and action recognition. Below are the supported datasets along with their download links:

### Object Recognition:
1. **N-MNIST**: [Dataset Link](https://www.garrickorchard.com/datasets/n-mnist)
2. **N-Cars**: [Dataset Link](https://www.prophesee.ai/2018/03/13/dataset-n-cars/)
3. **N-Caltech101**: [Dataset Link](https://www.garrickorchard.com/datasets/n-caltech101)
4. **CIFAR10-DVS**: [Dataset Link](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2?file=7713487)

### Action Recognition:
1. **ASL-DVS**: [Dataset Link](https://github.com/PIX2NVS/NVS2Graph)
2. **DVS-Gesture**: [Dataset Link](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system)

To use these datasets, download them using the provided links and place them in a known directory. The expected directory structure and additional configuration details can be found in `config/dataset_config.py`. In this configuration, the datasets are placed in the relative directory `../Datasets/` as shown in `config/dataset_config.py`. Then, a CSV file is required to properly load each dataset's `train` and `test` splits. These can be generated using `gen_dataset_csv.py` script, where the script is run from the command line, accepting arguments to specify the dataset path, name, and other options related to data splits and output formatting.

Example command:
```
python gen_dataset_csv.py --dataset_path /path/to/dataset --dataset_name MyDataset --data_split
```

For convenience, we provide pre-generated CSV files for train and test splits of these datasets in the `dataset_splits` folder. These CSV files are used to load the datasets correctly and were utilized to generate the results presented in our publication. Simply copy the relevant CSV files into the root directory of each dataset.

If you're using the DVS-Gesture dataset, initial processing is required with the following scripts:
1. `split_dvs_gesture_seqs.py` to split the recorded event sequences into sub-sequences per gesture.
2. `split_seqs_to_samples.py` to further split these sub-sequences into fixed-duration samples.

Finally, generate a CSV file for the DVS-Gesture dataset with `gen_dataset_csv.py`, or directly use the provided CSV files for this dataset by following the instructions above where step 2 must be run with the following parameters of `window size = 500 ms` and `step size = 250 ms`.

## Adding Custom Datasets

To integrate a custom dataset into this framework, follow these steps for seamless integration and usage:

1. **Prepare Your Dataset**: Ensure your dataset is organized in a format that is compatible with existing data loaders, or create a new data loader script within `scripts/datasets`.

2. **Create a Dataset Object**: Implement a new Python class for your dataset by inheriting from the `EventsDataset` class found in `scripts/datasets/Dataset.py`. In this class, you'll need to:
   - Call the `EventsDataset` initializer in your class's `__init__` method and pass the required parameters such as `csv_file`, `root_dir`, `event_rep`, `channels`, etc.
   - Define dataset-specific attributes such as `dataset_name`, `height`, `width`, `num_classes`, and `classes` (a list of class names).
   - Implement the `get_event_frame()` method to specify how a dataset's sample is read. This involves loading event data from files (based on the dataset's format) and converting it to the desired event representation format. Utilize appropriate data loading scripts from `scripts/data_processing/` based on your dataset's file format.

   For example, here's a template based on the `CIFAR10.py` dataset class:

   ```python
   from scripts.datasets.Dataset import EventsDataset
   from scripts.data_processing.load_atis_data import dat2mat # Or any other relevant loading script
   from scripts.data_processing.process_events import generate_event_representation

   class MyCustomDataset(EventsDataset):
       def __init__(self, csv_file, root_dir, event_rep='cstr', channels=1, split='train', delta_t=0, cache_dataset=False, transform=None, keep_size=False, frame_size=(128, 128), cache_transforms=False, augmentation_config=None, use_mp=True):
           super().__init__(csv_file, root_dir, event_rep, channels, split, delta_t, cache_dataset, transform, keep_size, frame_size, cache_transforms, augmentation_config, use_mp)
           
           self.dataset_name = 'MyCustomDataset'
           self.height = 128  # Original data's frame dimensions
           self.width = 128
           self.num_classes = 10  # Number of classes
           self.classes = ['class1', 'class2', ...]  # Label of each class
           
           assert len(self.classes) == self.num_classes
           self.class_index = self.dataset_data['class_index']

           if self.cache_dataset:
               self.cache_data()

       def get_event_frame(self, index):
           events_path = self.dataset_data['events_file_path'][index]
           events_file_path = os.path.join(self.root_dir, events_path)
           
           events_dict = dat2mat(events_file_path)  # Modify based on your dataset format
           if self.split == 'train':
               self.apply_events_augmentation(events_dict)

           event_frame = generate_event_representation(events_dict, self.width, self.height, delta_t=self.delta_t, representation=self.event_rep, channels=self.channels)
           return event_frame

3. **Add Dataset Configuration**:
    - In `config/dataset_config.py`, specify your dataset's configurations including paths (`root`, `train_csv`, `test_csv`) and any other relevant settings like spatial augmentation parameters.
    - Add your dataset's paths to the `dataset_config` dictionary.
    - Then, register your dataset object (created in step 2) by adding its name and reference to the `dataset_functions` dictionary.
      
5. **Generate CSV Files**: Utilize the gen_dataset_csv.py script to generate CSV files that map your dataset's samples to their corresponding labels and splits, if applicable. This step is crucial for enabling the loading and processing of the custom dataset during training and evaluation.

6. **Update and Run train.py Script**: Modify this script to use your custom dataset as an argument by setting the configuration's object `dataset` parameter to your dataset's name.

By following these steps, your custom dataset will be fully integrated into the existing framework, allowing for its utilization in training and evaluation processes alongside the supported datasets.


## Configuration

The `config.py` script plays a critical role in the training and evaluation processes by allowing users to customize various aspects of model training, evaluation, and data processing. 

### Key Configuration Options

- **Dataset and Event Representation**: Specify the dataset (`dataset`), event representation method (`event_rep`), and the number of input channels (`channels`).
- **Model Architecture and Pretraining**: Choose the classification model (`classifier`) and whether to use pre-trained weights (`pretrained_classifier`).
- **Data Preprocessing**: Set the batch size (`batch_size`), frame size for event data (`frame_size`), and whether to keep the original dataset size (`keep_size`).
- **Normalization and Augmentation**: Configure normalization parameters (`normalization`) and augmentation settings (`augmentation_config`).
- **Data Caching**: Enable caching for the training dataset (including both training and validation splits) and their transformations (`cache_dataset`, `cache_transforms`) to accelerate the training and evaluation process, with an option to cache the test set as well (`cache_test_set = True`). This approach is particularly useful when training multiple networks on the same dataset.
  - **Note**: Caching leverages local storage (`CACHE_TO_DISK` flag) by storing processed event samples as files, which can significantly speed up data loading, especially for large datasets and when using multiple CPU cores, by reducing interprocess communication overhead. Ensure sufficient disk space is available for caching, as it requires temporary storage that will be cleared post-process. If disk space is a concern, consider disabling `CACHE_TO_DISK` to avoid using storage, though this may slow down the process due to increased interprocess communication.


- **Output Management**: Define paths for saving results (`save_results`), models, and other outputs.
- **Computation Device**: Automatically selects the computation device based on CUDA availability (`DEVICE`).

Additional configurable parameters include training hyperparameters like learning rate, number of epochs, early stopping threshold, and more. Users can also specify paths for output directories such as models, checkpoints, and training history.

### Customizing Configurations

To customize the training and evaluation process, edit the relevant attributes in the `Configs` class within `config.py`. This script provides a centralized and structured way to manage all settings related to the training environment, model specifications, and data handling strategies.



## Citation
If you find our work useful in your research or use this code in an academic context, please cite the following:

```
@article{elshair2023cstr,
  title={CSTR: A Compact Spatio-Temporal Representation for Event-Based Vision},
  author={El Shair, Zaid A and Hassani, Ali and Rawashdeh, Samir A},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE},
  pages={102899-102916},
  doi={10.1109/ACCESS.2023.3316143}
}
```

***
Additionally, if you use any of the supported datasets, please cite their corresponding publications as listed below:

- **ASL-DVS**:
```
@inproceedings{bi2019graph,
title={Graph-based Object Classification for Neuromorphic Vision Sensing},
author={Bi, Y and Chadha, A and Abbas, A and Bourtsoulatze, E and Andreopoulos, Y},
booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
year={2019},
organization={IEEE}
}
```
- **CIFAR10-DVS**:
```
@article{li2017cifar10,
  title={Cifar10-dvs: an event-stream dataset for object classification},
  author={Li, Hongmin and Liu, Hanchao and Ji, Xiangyang and Li, Guoqi and Shi, Luping},
  journal={Frontiers in neuroscience},
  volume={11},
  pages={244131},
  year={2017},
  publisher={Frontiers},
  doi={10.3389/fnins.2017.00309},
  issn={1662-453X}
}
```

- **DVS-Gesture**:
```
@inproceedings{amir2017low,
  title={A low power, fully event-based gesture recognition system},
  author={Amir, Arnon and Taba, Brian and Berg, David and Melano, Timothy and McKinstry, Jeffrey and Di Nolfo, Carmelo and Nayak, Tapan and Andreopoulos, Alexander and Garreau, Guillaume and Mendoza, Marcela and others},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7243--7252},
  year={2017}
}
```

- **N-Caltech101** or **N-MNIST**:
```
@article{orchard2015converting,
  title={Converting static image datasets to spiking neuromorphic datasets using saccades},
  author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
  journal={Frontiers in neuroscience},
  volume={9},
  pages={159859},
  year={2015},
  publisher={Frontiers}
}
```

- **N-Cars**:
```
@inproceedings{sironi2018hats,
  title={HATS: Histograms of averaged time surfaces for robust event-based object classification},
  author={Sironi, Amos and Brambilla, Manuele and Bourdis, Nicolas and Lagorce, Xavier and Benosman, Ryad},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1731--1740},
  year={2018}
}
```

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.

### Contact
For any queries, please open an issue on the GitHub repository or contact the authors directly at zelshair@umich.edu for any questions.
