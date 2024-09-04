<div align="center">

# DIMIS - Domain Influence in Medical Image Segmentation

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![Conference](http://img.shields.io/badge/MLMI-2024-4b44ce.svg)](https://sites.google.com/view/mlmi2024/home)

</div>

## Description

Exploring the influence of different medical image representation towards segmentation.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/rauschecker-sugrue-labs/dimis
cd dimis

# [OPTIONAL] create conda environment
conda create -n dimis python=3.11
conda activate dimis

# install requirements
pip install --upgrade pip
pip install --upgrade setuptools
pip install -e .
```

#### Conda

```bash
# clone project
git clone https://github.com/rauschecker-sugrue-labs/dimis
cd dimis

# create conda environment and install dependencies
conda env create -f environment.yaml -n dimis

# activate conda environment
conda activate dimis
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
