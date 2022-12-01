# Stepwise Feature Fusion: Local Guides Global
This is the official implementation for [Stepwise Feature Fusion: Local Guides Global](https://arxiv.org/abs/2203.03635)

![SSformer](/images/ssformer.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=stepwise-feature-fusion-local-guides-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=stepwise-feature-fusion-local-guides-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=stepwise-feature-fusion-local-guides-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=stepwise-feature-fusion-local-guides-global)

## packages
- Please see requirements.txt

## Dataset
- The dataset we used can be download from [here](https://drive.google.com/file/d/1z48bsJftdp4akAlWOziqt6032huYYN9k/view?usp=sharing)

### Checkpoints
- The checkpoint for ssformer-S can be downloaded from [here](https://drive.google.com/file/d/1CdX0K1_ZDMrEVGK2cmBfp33lYxLEBwlw/view?usp=sharing)
- The checkpoint for ssformer-L can be downloaded from [here](https://drive.google.com/file/d/1CEwUOPm1otoEGfXSvcX-y1x80583-Q9C/view?usp=sharing)

## Usage
### Test
1. modified `configs/ssformer-S.yaml`
   - `dataset` set to your data path
   - `test.checkpoint_save_path` : path to your downloaded checkpoint
2. run `python test.py configs/ssformer-S.yaml`

### Train
1. modified `configs/train.yaml`
   - `model.pretrained_path` : mit pre-trained checkpoint path
   - `other` : path to save your training checkpoint and log file
2. run `python train.py configs/train.yaml`

## Citation
```
@InProceedings{10.1007/978-3-031-16437-8_11,
author="Wang, Jinfeng
and Huang, Qiming
and Tang, Feilong
and Meng, Jia
and Su, Jionglong
and Song, Sifan",
editor="Wang, Linwei
and Dou, Qi
and Fletcher, P. Thomas
and Speidel, Stefanie
and Li, Shuo",
title="Stepwise Feature Fusion: Local Guides Global",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="110--120",
abstract="Colonoscopy, currently the most efficient and recognized colon polyp detection technology, is necessary for early screening and prevention of colorectal cancer. However, due to the varying size and complex morphological features of colonic polyps as well as the indistinct boundary between polyps and mucosa, accurate segmentation of polyps is still challenging. Deep learning has become popular for accurate polyp segmentation tasks with excellent results. However, due to the structure of polyps image and the varying shapes of polyps, it is easy for existing deep learning models to overfit the current dataset. As a result, the model may not process unseen colonoscopy data. To address this, we propose a new state-of-the-art model for medical image segmentation, the SSFormer, which uses a pyramid Transformer encoder to improve the generalization ability of models. Specifically, our proposed Progressive Locality Decoder can be adapted to the pyramid Transformer backbone to emphasize local features and restrict attention dispersion. The SSFormer achieves state-of-the-art performance in both learning and generalization assessment.",
isbn="978-3-031-16437-8"
}

```
