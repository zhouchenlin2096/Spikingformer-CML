# Enhancing the Performance of Transformer-based Spiking Neural Networks by SNN-optimized Downsampling with Precise Gradient Backpropagation, [Arxiv 2023](https://arxiv.org/abs/2305.05954)
Our models achieve the state-of-the-art performance on several datasets (eg. **77.64 %** on ImageNet, **96.04 %** on CIFAR10, **80.37 %** on CIFAR100, **81.4% on** CIFAR10-DVS) in directly trained SNNs in 2023/05. Note: Our model achieves **78.46 %** on ImageNet with 288 * 288 resolution.

## News
[2022.8.18] Update trained models.

<br>
<br>
<p align="center">
<img src="https://github.com/zhouchenlin2096/Spikingformer-CML/blob/master/imgs/SNN-optimized-downsampling.png">
</p>

## Reference
If you find this repo useful, please consider citing:
```
@misc{zhou2023enhancing,
      title={Enhancing the Performance of Transformer-based Spiking Neural Networks by Improved Downsampling with Precise Gradient Backpropagation}, 
      author={Chenlin Zhou and Han Zhang and Zhaokun Zhou and Liutao Yu and Zhengyu Ma and Huihui Zhou and Xiaopeng Fan and Yonghong Tian},
      year={2023},
      eprint={2305.05954},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

## Main results on ImageNet-1K

| Model                     | Resolution| T     |  Param.     |Top-1 Acc| Download |
| :---                     | :---:     | :---: | :---:        |:---:    | :---:    |
| CML + Spikformer-8-384    | 224x224   | 4     |  16.81M     |72.73    |     -    |
| CML + Spikformer-8-512    | 224x224   | 4     |  29.68M     |75.61    |     -    |
| CML + Spikformer-8-768    | 224x224   | 4     |  66.34M     |77.34    |     -    |
| CML + Spikingformer-8-384 | 224x224   | 4     |  16.81M     |74.35    |     -    |
| CML + Spikingformer-8-512 | 224x224   | 4     |  29.68M     |76.54    |     -    |
| CML + Spikingformer-8-768 | 224x224   | 4     |  66.34M     |77.64    | [here](https://pan.baidu.com/s/1uTq6aPMknwb2PjDZ3J4g5g) |
| CML + Spikingformer-8-768 | 288x288   | 4     |  66.34M     |78.46    |     -    |

Download password: abcd

## Main results on CIFAR10/CIFAR100

| Model                      | T      |  Param.     | CIFAR10 Top-1 Acc |CIFAR100 Top-1 Acc|
| :---                      | :---:  | :---:       |  :---:            |:---: |
| CML + Spikformer-4-256     | 4      |  4.15M      | 94.82             |77.64  |
| CML + Spikformer-2-384     | 4      |  5.76M      | 95.63             |78.75  |
| CML + Spikformer-4-384     | 4      |  9.32M      | 95.93             |79.65  |
| CML + Spikformer-4-384-400E  | 4         |  9.32M | 96.04             |80.02  |
| CML + Spikingformer-4-256  | 4      |  4.15M      | 94.94             |78.19  |
| CML + Spikingformer-2-384  | 4      |  5.76M      | 95.54             |78.87  |
| CML + Spikingformer-4-384  | 4      |  9.32M      | 95.81             |79.98  |
| CML + Spikingformer-4-384-400E  | 4      |  9.32M     | 95.95         |80.37  |

## Main results on CIFAR10-DVS/DVS128

| Model                     | T      |  Param.     |  CIFAR10 DVS Top-1 Acc  | DVS 128 Top-1 Acc|
| :---                     | :---:  | :---:       | :---:                   |:---:             |
| CML + Spikformer-2-256    | 10     |  2.57M      | 79.2                    | 97.6             |
| CML + Spikformer-2-256    | 16     |  2.57M      | 80.9                    | 98.6             |
| CML + Spikingformer-2-256 | 10     |  2.57M      | 80.5                    | 97.2             |
| CML + Spikingformer-2-256 | 16     |  2.57M      | 81.4                    | 98.6             |


## Requirements
timm==0.6.12; cupy==11.3.0; torch==1.14.0+cu116; spikingjelly==0.0.0.0.12; pyyaml;

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

## Acknowledgement & Contact Information
Related project: [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer), [spikformer](https://github.com/ZK-Zhou/spikformer), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact zhouchl@pcl.ac.cn or zhouchenlin19@mails.ucas.ac.cn.
