# SOTA multi-label classifcation pipeline with Transformer backbone and Transformer head
The official SOTA classifcation pipeline of our project

## Requirements 
* [Pytorch](https://pytorch.org/)
* [Sklearn](https://scikit-learn.org/stable/)


## Installation
Please install pretrained backbone (.pth file) and put it under /pretrained_dir if you want to use pretrained backbone for your project. Support backbone:
* [Swin Transformer large_patch4_window12_384](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)
## Quick start

### Training

#### OLIVES
Please check the configs/olives/SwinTrans_Transformer_olives.py to set the necessary paths for your machine
```
python tools/train.py configs/olives/SwinTrans_Transformer_olives.py
```


### Testing

#### OLIVES
```
!python tools/test1.py config-path pretrained-file-path --out output-name.pkl
```
E.g:
```
!python tools/test1.py configs/olives/SwinTrans_Transformer_olives.py work_dirs/SwinTrans_Transformer_DB_OLIVES/epoch_2.pth --out OLIVES_result.pkl
```

## Pre-trained models


## Datasets



### Use our dataset


### Try your own

## License and Citation

## TODO


## Contact

