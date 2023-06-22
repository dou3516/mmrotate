# Introduction
This repo is forked from [openmmlab/mmrotate](https://github.com/open-mmlab/mmrotate).

## TODO
- [x] TTA
- [ ] New networks

## New features
- 2023.6.22: disable muliple scales in tta.py. It's unstable and need more finetune.
- 2023.6.19: add TTA from [zytx121's pull](https://github.com/open-mmlab/mmrotate/pull/771)
- 2023.6.1: add support CBA2023 Track: [细粒度密集船只目标检测任务](https://www.datafountain.cn/competitions/635) and Track: [基于亚米级影像的精细化目标检测](https://www.datafountain.cn/competitions/637)

## Get started

Refer to [official guide](https://mmrotate.readthedocs.io/en/1.x/get_started.html).

### Prerequisites
MMRotate works on Linux and Windows. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.6+

Step 0. Download and install Miniconda from the official website.
Step 1. Create a conda environment and activate it.

```sh
conda create mmrotate-1.x-base python=3.9
conda activate mmrotate-1.x-base
```

Step 2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```sh
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### Installation

Step 0. Install MMEngine and MMCV using MIM.

```sh
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"`
```

Step 1. Install MMDetection as a dependency.

```sh
mim install 'mmdet>=3.0.0rc2'
```

Step 2. Install MMRotate.

```sh
git clone https://github.com/dou3516/mmrotate.git -b 1.x
# "-b dev-1.x" means checkout to the `dev-1.x` branch.
cd mmrotate
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Data prepare

### Track: [细粒度密集船只目标检测任务](https://www.datafountain.cn/competitions/635)
Data directory should be like:
```sh
└── data
    └──2023_CBAC_ship_r1
        ├── test
        │   └── images
        └─── trainval
            ├── images
            └── annfiles
```

### Track: [基于亚米级影像的精细化目标检测](https://www.datafountain.cn/competitions/637)
```sh
└── data
    └──2023_CBAC_planeship_r1
        ├── test
        │   └── images
        └─── trainval
            ├── images
            └── annfiles
```

## Usage
### Track: [细粒度密集船只目标检测任务](https://www.datafountain.cn/competitions/635)
#### Train
```sh
# rotated_rtmdet, 12 epochs score 0.93+
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/rotated_rtmdet/rotated_rtmdet_l-1x-shiprs133_1024.py
# oriented_rcnn, 12 epochs score 0.94+, with more epochs 0.95+, with TTA 0.96+
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_shiprs133_1024.py
```

#### Test
```sh
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_shiprs133_1024.py work_dirs/oriented-rcnn-le90_swin-tiny_fpn_1x_shiprs133_1024/epoch_12.pth
```
Test with TTA (add --tta):
```sh
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_shiprs133_1024.py work_dirs/oriented-rcnn-le90_swin-tiny_fpn_1x_shiprs133_1024/epoch_12.pth --tta
```
#### Format for submission
```sh
# config submission file path in tools_dou/test2submit.py
python tools_dou/test2submit.py
```

### Track: [基于亚米级影像的精细化目标检测](https://www.datafountain.cn/competitions/637)
#### Train
```sh
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_planeship98_1024.py
```
#### Test for submission
```sh
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_planeship98_1024.py work_dirs/oriented-rcnn-le90_swin-tiny_fpn_1x_planeship98_1024/epoch_12.pth
```
#### Format for submission
```sh
# config submission file path in tools_dou/test2submit_planeship98.py
python tools_dou/test2submit_planeship98.py
```

### Attentions
- default batch_size=8
- BN in configs is 'BN'


# Below is original README of openmmlab/mmrotate

<div align="center">
  <img src="resources/mmrotate-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmrotate)](https://pypi.org/project/mmrotate)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmrotate.readthedocs.io/en/1.x/)
[![badge](https://github.com/open-mmlab/mmrotate/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrotate/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrotate/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrotate)
[![license](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)

[📘Documentation](https://mmrotate.readthedocs.io/en/1.x/) |
[🛠️Installation](https://mmrotate.readthedocs.io/en/1.x/install.html) |
[👀Model Zoo](https://mmrotate.readthedocs.io/en/1.x/model_zoo.html) |
[🆕Update News](https://mmrotate.readthedocs.io/en/1.x/notes/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmrotate/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmrotate/issues/new/choose)

</div>

<!--中/英 文档切换-->

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

MMRotate is an open-source toolbox for rotated object detection based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.6+**.

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

<details open>
<summary><b>Major Features</b></summary>

- **Support multiple angle representations**

  MMRotate provides three mainstream angle representations to meet different paper settings.

- **Modular Design**

  We decompose the rotated object detection framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Strong baseline and State of the art**

  The toolbox provides strong baselines and state-of-the-art methods in rotated object detection.

</details>

## What's New

### Highlight

We are excited to announce our latest work on real-time object recognition tasks, **RTMDet**, a family of fully convolutional single-stage detectors. RTMDet not only achieves the best parameter-accuracy trade-off on object detection from tiny to extra-large model sizes but also obtains new state-of-the-art performance on instance segmentation and rotated object detection tasks. Details can be found in the [technical report](https://arxiv.org/abs/2212.07784). Pre-trained models are [here](configs/rotated_rtmdet).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

**v1.0.0rc1** was released in 30/12/2022:

- Support [RTMDet](configs/rotated_rtmdet) rotated object detection models. The technical report of RTMDet is on [arxiv](https://arxiv.org/abs/2212.07784)
- Support [H2RBox](configs/h2rbox) models. The technical report of H2RBox is on [arxiv](https://arxiv.org/abs/2210.06742)

## Installation

Please refer to [Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) for more detailed instruction.

## Getting Started

Please see [Overview](https://mmrotate.readthedocs.io/en/1.x/overview.html) for the general introduction of MMRotate.

For detailed user guides and advanced guides, please refer to our [documentation](https://mmrotate.readthedocs.io/en/1.x/):

- User Guides
  - [Train & Test](https://mmrotate.readthedocs.io/en/1.x/user_guides/index.html#train-test)
    - [Learn about Configs](https://mmrotate.readthedocs.io/en/1.x/user_guides/config.html)
    - [Inference with existing models](https://mmrotate.readthedocs.io/en/1.x/user_guides/inference.html)
    - [Dataset Prepare](https://mmrotate.readthedocs.io/en/1.x/user_guides/dataset_prepare.html)
    - [Test existing models on standard datasets](https://mmrotate.readthedocs.io/en/1.x/user_guides/train_test.html)
    - [Train predefined models on standard datasets](https://mmrotate.readthedocs.io/en/1.x/user_guides/train_test.html)
    - [Test Results Submission](https://mmrotate.readthedocs.io/en/1.x/user_guides/test_results_submission.html)
  - [Useful Tools](https://mmrotate.readthedocs.io/en/1.x/user_guides/index.html#useful-tools)
- Advanced Guides
  - [Basic Concepts](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#basic-concepts)
  - [Component Customization](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#component-customization)
  - [How to](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#how-to)

We also provide colab tutorial [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo/MMRotate_Tutorial.ipynb).

To migrate from MMRotate 0.x, please refer to [migration](https://mmrotate.readthedocs.io/en/1.x/migration.html).

## Model Zoo

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [Model Zoo](docs/en/model_zoo.md) page.

<details open>
<summary><b>Supported algorithms:</b></summary>

- [x] [Rotated RetinaNet-OBB/HBB](configs/rotated_retinanet/README.md) (ICCV'2017)
- [x] [Rotated FasterRCNN-OBB](configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
- [x] [Rotated RepPoints-OBB](configs/rotated_reppoints/README.md) (ICCV'2019)
- [x] [Rotated FCOS](configs/rotated_fcos/README.md) (ICCV'2019)
- [x] [RoI Transformer](configs/roi_trans/README.md) (CVPR'2019)
- [x] [Gliding Vertex](configs/gliding_vertex/README.md) (TPAMI'2020)
- [x] [Rotated ATSS-OBB](configs/rotated_atss/README.md) (CVPR'2020)
- [x] [CSL](configs/csl/README.md) (ECCV'2020)
- [x] [R<sup>3</sup>Det](configs/r3det/README.md) (AAAI'2021)
- [x] [S<sup>2</sup>A-Net](configs/s2anet/README.md) (TGRS'2021)
- [x] [ReDet](configs/redet/README.md) (CVPR'2021)
- [x] [Beyond Bounding-Box](configs/cfa/README.md) (CVPR'2021)
- [x] [Oriented R-CNN](configs/oriented_rcnn/README.md) (ICCV'2021)
- [x] [GWD](configs/gwd/README.md) (ICML'2021)
- [x] [KLD](configs/kld/README.md) (NeurIPS'2021)
- [x] [SASM](configs/sasm_reppoints/README.md) (AAAI'2022)
- [x] [Oriented RepPoints](configs/oriented_reppoints/README.md) (CVPR'2022)
- [x] [KFIoU](configs/kfiou/README.md) (ICLR'2023)
- [x] [H2RBox](configs/h2rbox/README.md) (ICLR'2023)
- [x] [PSC](configs/psc/README.md) (CVPR'2023)
- [x] [RTMDet](configs/rotated_rtmdet/README.md) (arXiv)

</details>

## Data Preparation

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## FAQ

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMRotate. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We appreciate the [Student Innovation Center of SJTU](https://www.si.sjtu.edu.cn/) for providing rich computing resources at the beginning of the project. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {7331–7334},
  numpages = {4},
  year={2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
