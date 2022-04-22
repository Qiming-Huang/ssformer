# Stepwise Feature Fusion: Local Guides Global
This is the official implementation for [Stepwise Feature Fusion: Local Guides Global](https://arxiv.org/abs/2203.03635)

![SSformer](/images/ssformer.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=stepwise-feature-fusion-local-guides-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=stepwise-feature-fusion-local-guides-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=stepwise-feature-fusion-local-guides-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-feature-fusion-local-guides-global/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=stepwise-feature-fusion-local-guides-global)

## Dataset
- The dataset we used can be download from [here](https://drive.google.com/file/d/1z48bsJftdp4akAlWOziqt6032huYYN9k/view?usp=sharing)

## Result and Checkpoint
### ssformer-S
|  dataset   | meanDic  | meanIou  | wFm  | Sm  | meanEm  | mae  |
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  ----  | 
| CVC-300  | 0.887 | 0.821  | 0.869 | 0.929  | 0.000 | 0.007  |
| CVC-ClinicDB  | 0.916 | 0.873  | 0.924 | 0.937  | 0.000 | 0.007  |
| Kvasir  | 0.925 | 0.878  | 0.921 | 0.931  | 0.000 | 0.017  |
| CVC-ColonDB  | 0.772 | 0.697  | 0.766 | 0.844  | 0.000 | 0.036  |
| ETIS | 0.767 | 0.698  | 0.736 | 0.863  | 0.000 | 0.016 |
### ssformer-L
|  dataset   | meanDic  | meanIou  | wFm  | Sm  | meanEm  | mae  |
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  ----  | 
| CVC-300  | 0.895 | 0.827  | 0.881 | 0.933  | 0.000 | 0.007  |
| CVC-ClinicDB  | 0.906 | 0.855  | 0.913 | 0.929  | 0.000 | 0.008 |
| Kvasir  | 0.917 | 0.864  | 0.916 | 0.922  | 0.000 | 0.022  |
| CVC-ColonDB  | 0.802 | 0.721 | 0.798 | 0.860  | 0.000 | 0.031  |
| ETIS | 0.796 | 0.720  | 0.771 | 0.873  | 0.000 | 0.014 |

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
@article{wang2022stepwise,
  title={Stepwise Feature Fusion: Local Guides Global},
  author={Wang, Jinfeng and Huang, Qiming and Tang, Feilong and Meng, Jia and Su, Jionglong and Song, Sifan},
  journal={arXiv preprint arXiv:2203.03635},
  year={2022}
}
```
