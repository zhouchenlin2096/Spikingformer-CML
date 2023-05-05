# Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network, [Arxiv 2023](https://arxiv.org/abs/2304.11954)
Spikingformer have achieved 77.64% in ImageNet 2012, 80.75% in CIFAR100, 95.94% in CIFAR10, 81.4% in CIFAR10-DVS, 98.6% in DVS Guesture, achieving the best results in all the above five datasets. 
We have built a novel and pure spiking neural network model based on transformer, named Spikingformer, which is the state-of-the-art in directly trained SNNs models. 
Paper and code are coming soon !

## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Zhang, Han and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023}
  url={https://arxiv.org/abs/2304.11954}
}
```
Our codes are based on the official imagenet example by PyTorch, pytorch-image-models by Ross Wightman and SpikingJelly by Wei Fang.

<p align="center">
<img src="https://github.com/.png">
</p>

### Requirements
timm==0.5.4

cupy==10.3.1

pytorch==1.10.0+cu111

spikingjelly==0.0.0.0.12

pyyaml

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


### Training  on ImageNet
Setting hyper-parameters in imagenet.yml

```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Testing ImageNet Val data 
```
cd imagenet
python test.py
```

### Training  on cifar10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```
### Training  on cifar10DVS
```
cd cifar10dvs
python train.py
```

