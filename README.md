# FFA

**A deep learning-based fundus fluoresceinangiography imageanalytics protocowith classification and segmentation tasks**
<p align="center">
    <img src="Figures/abstract.png" title="Abstract" width="500" /> 
</p>
## Getting Started

### Requirements

* **Ubuntu 16.04 or higher, Windows 10 or higher with Anaconda or Mini-conda**
* **Python 3.7**
* **PyTorch = 1.11.0**
* **tensorflow-gpu = 1.14.0**
* **CUDA 11.3**
* **More packages please refer to [`requirement`](https://github.com/huapu4/FFA/blob/main/requirement.txt)**

### Installation

a. Create a conda virtual environment and activate it, e.g.,

```
conda create --name [your_name] python=3.7
conda activate [your_name]
```

b. Clone the FFA repository.

```
git clone https://github.com/huapu4/FFA.git
cd FFA
```

c. Install the environments.

```python 
pip install -r requirement.txt
```

### Data preparation

**There will be three datasets to be prepared, respectively in
folder: [01.Phase_identification](https://github.com/huapu4/FFA/tree/main/01.Phase_identification/dataset)
, [02.Disease_diagnosis](https://github.com/huapu4/FFA/tree/main/02.Disease_diagnosis/dataset)
, [03.Area_segmentation](https://github.com/huapu4/FFA/tree/main/03.Area_segmentation/FFA_dataset).**

a. Put your own data in [./origin_data](https://github.com/huapu4/FFA/tree/main/origin_data), stored according to
specifications.

```
cd origin_data
tree -d -L 2
.
├── 01.phase_identification
│   ├── arterial_phase
│   ├── non_ffa
│   └── venous_phase
├── 02.disease_diagnosis
│   ├── brvo
│   ├── none_np
│   ├── normal
│   └── with_np
└── 03.area_segmentation
    ├── labeled_data
    └── voc_data
```

b. For [01.Phase_identification](https://github.com/huapu4/FFA/tree/main/01.Phase_identification/dataset)
and [02.Disease_diagnosis](https://github.com/huapu4/FFA/tree/main/02.Disease_diagnosis/dataset), they are
classification
task, with the same data deployment.

```
cd FFA
python cls_data_deploy.py --task [name_of_task] --input [origin_data_folder] --output [output_folder] --prop [proportion of trainset]
```

c. For [03.Area_segmentation](https://github.com/huapu4/FFA/tree/main/03.Area_segmentation/FFA_dataset), it is a
segmentation task, please run.
```
python seg_data_deploy.py --input [VOC_data_folder] --output [output_folder] --prop [proportion of trainset]
```

### Training and evaluation


