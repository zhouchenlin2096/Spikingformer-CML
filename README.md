# Enhancing the Performance of Transformer-based Spiking Neural Networks by Improved Downsampling with Precise Gradient Backpropagation, [Arxiv 2023]()
Our models achieve the state-of-the-art performance on several datasets (eg. 77.64 % on ImageNet, 96.04 % on CIFAR10, 80.75 % on CIFAR10, 81.4% on CIFAR10-DVS) in directly trained SNNs in 2023/05.

## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Zhang, Han and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023},
  url={https://arxiv.org/abs/2304.11954}
}
```
Our codes are based on the official imagenet example by PyTorch, pytorch-image-models by Ross Wightman and SpikingJelly by Wei Fang.

## Main results on ImageNet-1K

| Model                     | Resolution| T     |  Param.     |Top-1 Acc|
| :---:                     | :---:     | :---: | :---:       |:---:    |
| CML + Spikingformer-8-384 | 224x224   | 4     |  16.81M     |74.35    |
| CML + Spikingformer-8-512 | 224x224   | 4     |  29.68M     |76.54    |
| CML + Spikingformer-8-768 | 224x224   | 4     |  66.34M     |77.64    |


## Main results on CIFAR10/CIFAR100

| Model                      | T      |  Param.     | CIFAR10 Top-1 Acc |CIFAR100 Top-1 Acc|
| :---:                      | :---:  | :---:       |  :---:            |:---: |
| CML + Spikingformer-4-256  | 4      |  4.15M      | 94.94             |78.19  |
| CML + Spikingformer-2-384  | 4      |  5.76M      | 95.54             |78.87  |
| CML + Spikingformer-4-384  | 4      |  9.32M      | 95.81             |79.98  |
| CML + Spikingformer-4-384-400E  | 4      |  9.32M     | 95.95         |80.75  |

## Main results on CIFAR10-DVS/DVS128

| Model                     | T      |  Param.     |  CIFAR10 DVS Top-1 Acc  | DVS 128 Top-1 Acc|
| :---:                     | :---:  | :---:       | :---:                   |:---:             |
| CML + Spikingformer-2-256 | 10     |  2.57M      | 80.5                    | 97.2             |
| CML + Spikingformer-2-256 | 16     |  2.57M      | 81.4                    | 98.6             |


## Requirements
timm==0.5.4;
cupy==10.3.1;
pytorch==1.10.0+cu111;
spikingjelly==0.0.0.0.12;
pyyaml;

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train
### Training  on ImageNet
Setting hyper-parameters in imagenet.yml

```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar10
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py
```

