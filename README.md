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

## Supported Datasets
This project supports the following event-based datasets out of the box:

### Object Recognition:
1. **N-MNIST** (link: https://www.garrickorchard.com/datasets/n-mnist)
2. **N-Cars** (link: https://www.prophesee.ai/2018/03/13/dataset-n-cars/)
3. **N-Caltech101** (link: https://www.garrickorchard.com/datasets/n-caltech101)
4. **CIFAR10-DVS** (link: https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2?file=7713487)

### Action Recognition:
1. **ASL-DVS** (link: https://github.com/PIX2NVS/NVS2Graph)
2. **DVS-Gesture** (link: https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system)

To use these datasets, use the links above to download them, then move to a Dataset/ directory. In this source code, the datasets are placed in the relative directory `../Datasets/` as shown in `config/dataset_config.py`. A .csv file is required to properly load a dataset for both the `train` and `test` splits. These can be generated using `gen_dataset_csv.py` script, where the script is run from the command line, accepting arguments to specify the dataset path, name, and other options related to data splits and output formatting.

Example command:
```
python gen_dataset_csv.py --dataset_path /path/to/dataset --dataset_name MyDataset --data_split
```
Alternatively, for the supported datasets above, we provide each dataset's CSV file used to generate the results in our paper in the CSVs folder. Copy each dataset's CSV files to the root of the dataset folder (indicated by each dataset's `root` parameter in `dataset_config.py`). Each dataset sample file must be available at the relative path indicated in these CSV files for each split. 

Note, when using the DVS-Gesture dataset, it must be initially processed by running:
1. split_dvs_gesture_seqs.py
2. split_seqs_to_samples.py

Then, you can generate a CSV file for this dataset using the `gen_dataset_csv.py` script as described above.

Otherwise, to use the provided dataset CSV, step 2 must be run with the following parameters of  `window size = 500 ms` and `step size = 250 ms`.

## Adding Custom Datasets
To add a custom dataset, follow these steps:
1. Prepare your dataset in a format compatible with the existing data loaders or create a new data loader script in scripts/datasets.
2. Add your dataset configuration in config/dataset_config.py. This includes paths, normalization parameters, and any other relevant settings.
3. Use gen_dataset_csv.py to generate CSV files that map your dataset's samples to their corresponding labels and splits (if applicable).
4. Modify train.py and evaluate.py scripts to accept your dataset as an argument or configure it directly in config/config.py.



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


Additionally, if you use any of the supported datasets, please cite their corresponding publications as listed below:

**ASL-DVS**:
```
@inproceedings{bi2019graph,
title={Graph-based Object Classification for Neuromorphic Vision Sensing},
author={Bi, Y and Chadha, A and Abbas, A and Bourtsoulatze, E and Andreopoulos, Y},
booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
year={2019},
organization={IEEE}
}
```
**CIFAR10-DVS**:
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

**DVS-Gesture**:
```
@inproceedings{amir2017low,
  title={A low power, fully event-based gesture recognition system},
  author={Amir, Arnon and Taba, Brian and Berg, David and Melano, Timothy and McKinstry, Jeffrey and Di Nolfo, Carmelo and Nayak, Tapan and Andreopoulos, Alexander and Garreau, Guillaume and Mendoza, Marcela and others},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7243--7252},
  year={2017}
}
```

**N-Caltech101** or **N-MNIST**:
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

**N-Cars**:
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
