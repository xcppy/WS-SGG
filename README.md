# Scene Graph Generation from Natural Language Supervision
This is our Pytorch implementation for the paper:
> Xingchen Li, Long Chen, Wenbo Ma, Yi Yang, and Jun Xiao. Integrating Object-aware and Interaction-aware Knowledge for Weakly Supervised Scene Graph Generation. In MM 2022.


## Contents

1. [Installation](#Installation)
2. [Data](#Data)
3. [Metrics](#Metrics)
4. [Pretrained Object Detector](#Pretrained-Object-Detector)
5. [Grounding Module](#Grounding-Module)
6. [Pretrained Scene Graph Generation Models](#Pretrained-Scene-Graph-Generation-Models)
7. [Model Training](#Model-Training)
8. [Model Evaluation](#Model-Evaluation)
9. [Acknowledgement](#Acknowledgement)
10. [Reference](#Reference)


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Data

Check [DATASET.md](DATASET.md) for instructions of data downloading.

## Metrics

Explanation of metrics in this toolkit are given in [METRICS.md](METRICS.md)

## Pretrained Object Detector

In this project, we primarily use the detector Faster RCNN pretrained on Open Images dataset. To use this repo, you don't need to run this detector. You can directly download the extracted detection features, as the instruction in [DATASET.md](DATASET.md). If you're interested in this detector, the pretrained model can be found in [TensorFlow 1 Detection Model Zoo: faster_rcnn_inception_resnet_v2_atrous_oidv4](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#open-images-trained-models). 

For fully supervised models, you can use the detector pretrained by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch#pretrained-models). You can download this [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ) and extract all the files to the directory `checkpoints/pretrained_faster_rcnn`.

## Grounding Module
The codes for grounding module can be found [here](https://github.com/ChopinSharp/ws-sgg-grounder).
We provide generated grounding results for [VG captions](https://drive.google.com/file/d/1jVmN_woqXQ2ovfXhvfVXzK71Dn6dwu6F/view?usp=sharing).

## Pretrained Scene Graph Generation Models

Our pretrained SGG models can be downloaded on [Google Drive](https://drive.google.com/file/d/1kjwmw1wdDnqwEsP__FrKVgiLj7rsUWxr/view?usp=sharing). The details of these models can be found in Model Training section below. After downloading, please put all the folders to the directory `checkpoints/`. 

## Model Training

To train our scene graph generation models, run the script
```
bash train.sh MODEL_TYPE
```
where `MODEL_TYPE` specifies the training supervision, the training dataset and the scene graph generation model. See details below.

1. VG caption supervised models: trained by image-text pairs in VG dataset
    * `VG_Caption_Ground_*`: train a SGG model with the generated pseudo labels by our methods. `*` represents the model name and can be `Motifs`, `Uniter`.
    * `VG_Caption_SGNLS_*`: train a SGG model with generated pseudo labels from detector. `*` represents the model name and can be `Motifs`, `Uniter`.

2. VG unlocal supervised models: trained by unlocalized scene graph labels
    * `Unlocal_VG_Ground_*`: train a SGG model with the generated pseudo labels by our methods.
    * `Unlocal_VG_SGNLS_*`: train a SGG model with the generated pseudo labels from detector.



You can set `CUDA_VISIBLE_DEVICES` in `train.sh` to specify which GPUs are used for model training (e.g., the default script uses 2 GPUs).

## Model Evaluation

To evaluate the trained scene graph generation model, you can reuse the commands in `train.sh` by simply changing `WSVL.SKIP_TRAIN` to `True` and setting `OUTPUT_DIR` as the path to your trained model. 

## Acknowledgement

This repository was built based on [SGG_from_NLS](https://github.com/YiwuZhong/SGG_from_NLS), [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) for scene graph generation and [UNITER](https://github.com/ChenRocks/UNITER) for image-text representation learning.


## Reference
If you find this project helps your research, please kindly consider citing our project or papers in your publications.
