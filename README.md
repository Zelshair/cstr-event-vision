# CSTR: A Compact Spatio-Temporal Representation for Event-Based Vision

This repository contains the source code for the paper "CSTR: A Compact Spatio-Temporal Representation for Event-Based Vision" published in IEEE Access, developed by Zaid El Shair. Our work introduces a novel representation for event-based vision, significantly enhancing the processing of event data with deep learning architectures.
![Graphical Abstract](https://github.com/Zelshair/cstr-event-vision/assets/50595418/3ec86fc6-84c2-4da0-8b0b-9c12852d4e27)
**Figure**: An event camera captures sparse and asynchronous events that represent brightness changes at each pixel in the frame. To process these events in batches and utilize modern computer vision architectures, an intermediate representation is required. In this work, we propose the compact spatio-temporal representation (CSTR) that encodes the event batch’s spatial, temporal, and polarity information in a 3-channel image-like format.

## Abstract

Event-based vision is a novel perception modality that offers several advantages, such as high dynamic range and robustness to motion blur. In order to process events in batches and utilize modern computer vision deep-learning architectures, an intermediate representation is required. Nevertheless, constructing an effective batch representation is non-trivial. In this paper, we propose a novel representation for event-based vision, called the compact spatio-temporal representation (CSTR). The CSTR encodes an event batch’s spatial, temporal, and polarity information in a 3-channel image-like format. It achieves this by calculating the mean of the events’ timestamps in combination with the event count at each spatial position in the frame. This representation shows robustness to motion-overlapping, high event density, and varying event-batch durations. Due to its compact 3-channel form, the CSTR is directly compatible with modern computer vision architectures, serving as an excellent choice for deploying event-based solutions. In addition, we complement the CSTR with an augmentation framework that introduces randomized training variations to the spatial, temporal, and polarity characteristics of event data. Experimentation over different object and action recognition datasets shows that the CSTR outperforms other representations of similar complexity under a consistent baseline. Further, the CSTR is made more robust and significantly benefits from the proposed augmentation framework, considerably addressing the sparseness in event-based datasets.

## Repository Structure
```
.
├── scripts
│ ├── data_processing # Scripts for preprocessing event data
│ ├── training # Training scripts for models
│ └── evaluation # Evaluation and testing scripts
├── datasets # Dataset configuration and processing utilities
├── utils # Utility functions and helpers
├── config # Configuration files for models, datasets, and training
└── README.md
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
## Supported Datasets
### Object Recognition:
1. **N-MNIST** (link: https://www.garrickorchard.com/datasets/n-mnist)
2. **N-Cars** (link: https://www.prophesee.ai/2018/03/13/dataset-n-cars/)
3. **N-Caltech101** (link: https://www.garrickorchard.com/datasets/n-caltech101)
4. **CIFAR10-DVS** (link: https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2?file=7713487)

### Action Recognition:
1. **ASL-DVS** (link: https://github.com/PIX2NVS/NVS2Graph)
2. **DVS-Gesture** (link: https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system)

Note: Other datasets can be added by:
1. Create 
2. add to dataset config
### downloading datasets
download each dataset

dataset csv used to generate the results in our paper are provided in CVSs folder. Copy each dataset's csv files to the root of the dataset folder. Each dataset sample file must be available at the relative path indicated in these csv files for each split.

Note, for DVS Gesture dataset, it must be first processed by running:
1. script 1
2. script 2 (with 500 ms and 250 ms)
The generated csv files can then be used.

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
