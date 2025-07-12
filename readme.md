# Multi-Task Attention Network (MTAN) for NYUv2 Scene Understanding  
**Deep Learning Coursework – MSc Artificial Intelligence**

## Overview

This project is part of the final coursework for the **Deep Learning** module in the MSc in Artificial Intelligence. The goal of the project is to study, reimplement, and experiment with a state-of-the-art **Multi-Task Learning (MTL)** architecture called **Multi-Task Attention Network (MTAN)**.

Multi-Task Learning aims to improve generalization performance by training a single model to perform multiple related tasks jointly. By leveraging shared representations, MTL can reduce overfitting, enable transfer across tasks, and improve model efficiency.

### What is MTAN?

MTAN is an MTL architecture proposed in the paper:  
**"End-To-End Multi-Task Learning with Attention"**  
by Shikun Liu, Edward Johns, and Andrew J. Davison ([CVPR 2019](https://arxiv.org/abs/1803.10704))

Instead of sharing features naively, MTAN introduces **task-specific attention modules** that learn soft attention masks over a shared backbone. This allows the model to dynamically select relevant features for each task at each layer, achieving high task-specific performance while retaining parameter efficiency.

---

## Project Structure

- `src/segnet_mtan.py`: Custom PyTorch implementation of the MTAN architecture using a SegNet backbone.
- `train.py`: Training script supporting multi-task and single-task setups.
- `src/utils.py`: Helper functions for training and evaluation

---

## Training Instructions

Training is launched via `train.py` with multiple configurable options:

### Basic Command

```bash
python train.py --task all --weight dwa
```
### Arguments
Argument	Description	Default
--batch_size	Batch size for training	2
--epochs	Number of training epochs	20
--weight	Multi-task loss weighting: equal, uncert, dwa	equal
--dataroot	Path to the dataset root (expects NYUv2 format)	nyuv2
--temp	Temperature for Dynamic Weight Average (DWA)	2.0
--apply_augmentation	Apply data augmentation (flip, scale, etc.)	False
--downsample_ratio	Ratio to downsample images for augmentation	1.0
--resume	Resume training from last checkpoint	False
--task	Task to train: all, semantic, depth, normal	all
--single_task	Train a single-task model	False
--segnet	Train baseline SegNet instead of MTAN	False
--lambda_consistency	Weight for inter-task consistency loss (normals-depth)	0.0

## Dataset

The `NYUv2` dataset, pre-processed by the authors, can be found [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0).


## Supported Tasks
The model supports training on the NYUv2 dataset for the following tasks:
- Semantic Segmentation (13-class)
- Depth Estimation
- Surface Normal Prediction