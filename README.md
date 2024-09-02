<p align="center">
  <h1 align="center">Latent Fingerprint Matching via Dense Minutia Descriptor</h1>
  <p align="center">
    <strong>Zhiyu Pan</strong></a>
    &nbsp;&nbsp;
    <strong>Yongjie Duan</strong>
    &nbsp;&nbsp;
    <a href="https://xiongjunguan.github.io/"><strong>Xiongjun Guan</strong></a>
    <br>
    <a href="http://ivg.au.tsinghua.edu.cn/~jfeng/"><strong>JianJiang Feng*</strong></a>
    &nbsp;&nbsp;
    <strong>Jie Zhou</strong>
  </p>
  <br>
  <div align="center">
    <img src="./figures/dmd_illustration.png", width="700">
  </div>
  <p align="center">
    <a href="https://arxiv.org/abs/2405.01199"><img alt='arXiv' src='https://img.shields.io/badge/arXiv-2405.01199-b31b1b.svg'></a>
  </p>
  <br>
</p>

<!-- # Latent Fingerprint Matching via Dense Minutia Descriptor
This repository contains the code for the paper "Latent Fingerprint Matching via Dense Minutia Descriptor". -->


## Overview

The code in this repository implements the methods described in the paper to match latent fingerprints using dense minutia descriptors. This method aims to improve the accuracy and efficiency of latent fingerprint matching. This paper has been accepted by IJCB 2024.

## News and Update

* [July 3th] Inference code Release.
    * The basic inference code of DMD.  

## Requirements
It is recommended to run our code on a Nvidia GPU with a linux system. We have not yet tested on other configurations.

Basic requirements:
- Python==3.8
- torch==1.10.1
- scikit-learn==1.3.0
- scipy==1.10.1
- numpy==1.24.4

Download the './fptools', simply run the following commands to download to the root directory of this project.
```
git clone https://github.com/keyunj/fptools.git
```

## Download Weights

Download the [weights](https://cloud.tsinghua.edu.cn/f/fd5ca22af0eb44afa124/?dl=1) trained by NIST SD14, and place it at "./logs/DMD/".

## Prepare the Datasets
Place the evaluated dataset in the structure as follows:
```bash
TEST_DATA/
│
├── NIST27/                 # Replace to custom dataset
│   ├── image/              # image folder
│        ├── query/         # query images
│            ├── xxxx.bmp
            ......
│        ├── gallery/       # gallery images
│            ├── xxxx.bmp
            ......
│   ├── mnt/                # minutiae folder
│        ├── query/         # query minutiae files
│            ├── xxxx.mnt
            ......
│        ├── gallery/       # gallery minutiae files
│            ├── xxxx.mnt             

```
Run the script:
```
python dump_dataset_mnteval.py --prefix /path/to/dataset
```

## Evaluating
Run the script:
```
python evaluate_mnt.py
```

## License and Usage Restrictions
Code related to the DMD is under Apache 2.0 license. 
**This project is intended for academic research purposes only and cannot be used for commercial purposes.**

## Citation
If you find our repo helpful, please consider leaving a star or cite our paper :)
```
@article{pan2024latent,
  title={Latent Fingerprint Matching via Dense Minutia Descriptor},
  author={Pan, Zhiyu and Duan, Yongjie and Guan, Xiongjun and Feng, Jianjiang and Zhou, Jie},
  journal={arXiv preprint arXiv:2405.01199},
  year={2024}
}
```