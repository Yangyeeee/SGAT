# SGAT

This reposityory contains the PyTorch implementation for [Sparse Graph Attention Networks](https://arxiv.org/abs/1912.00552)



## Demo

The evolution of the graph of Zachary's Karate Club at different training epochs:

![demo](https://github.com/sndnyang/xvat/raw/master/demo/moons.gif)

## Requirements

    PyTorch >= 0.4.0
    tensorboardX <= 1.6.0
    numpy


​    
# Download Datasets

1. git clone https://github.com/sndnyang/xvat

2. Download the Datasets(MNIST, SVHN, CIFAR10):

   On CIFAR-10

   ```
   python dataset/cifar10.py --data_dir=./data/cifar10/
   ```

   On SVHN

   ```
   python dataset/svhn.py --data_dir=./data/svhn/
   ```

   Visualize on CIFAR-10 base on normalized data in the range between [0, 1], otherwise, you should not enable --vis

   ```
   python dataset/cifar10_0-1.py --data_dir=./data1.0/cifar10/
   ```
# Usage

## Basic Parameters

```
optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     mnist, cifar10, svhn (default: cifar10)
  --data-dir DATA_DIR   default: data
  --trainer TRAINER     ce, vat, inl0(L0, not ten), (default: inl0)
  --size SIZE           size of training data set, fixed size for datasets
                        (default: 4000)
  --arch ARCH           CNN9 for semi supervised learning on dataset
  --layer LAYER         the layer of CNN used by the generator, (default: 2)
  --num-epochs N        number of epochs (default: 100)
  --num-batch-it N      number of batch iterations (default: 400)
  --seed N              random seed (default: 1)
  --no-cuda             disables CUDA training
  --gpu-id N            gpu id list (default: auto select)
  --log-interval N      iterations to wait before logging status, (default: 1)
  --batch-size BATCH_SIZE
                        batch size of training data set, MNIST uses 100
                        (default: 32)
  --ul-batch-size UL_BATCH_SIZE
                        batch size of unlabeled data set, MNIST uses 250
                        (default: 128)
  --lr LR               learning rate (default: 0.001)
  --lr-a LR_A           learning rate for log_alpha (default: 0.001)
  --lr-decay LR_DECAY   learning rate decay used on MNIST (default: 0.95)
  --epoch-decay-start EPOCH_DECAY_START
                        start learning rate decay used on SVHN and cifar10
                        (default: 80)
  --alpha ALPHA         alpha for KL div loss (default: 1)
  --eps EPS             alpha for KL div loss (default: 1)
  --lamb LAMB           lambda for unlabeled l0 loss (default: 1)
  --lamb2 LAMB2         lambda for unlabeled smooth loss (default: 0)
  --l2-lamb L2_LAMB     lambda for L2 norm (default: )
  --zeta ZETA           zeta for L0VAT, always > 1 (default: 1.1)
  --beta BETA           beta for L0VAT (default: 0.66)
  --gamma GAMMA         gamma for L0VAT, always < 0 (default: -0.1)
  --affine              batch norm affine configuration
  --top-bn              enable top batch norm layer
  --ent-min             use entropy minimum
  --kl KL               unlabel loss computing, (default: 1)
  --aug-trans           data augmentation
  --aug-flip            data augmentation flip(for CIFAR)
  --drop DROP           dropout rate, (default: 0.5)
  --log-dir S           tensorboard directory, (default: an absolute path)
  --log-arg S           show the arguments in directory name
  --debug               compare log side by side
  --vis                 visual by tensor board
  -r S, --resume S      resume from pth file
```



## Supervised Learning

### mnist:

##### xAT

```
python train.py --dataset=cora --l0=1 --lr=0.01 --num-heads=2 --gpu=4 --num-layers=2 --epochs=200 --num-hidden=32 --loss_l0 1e-6 --sess cora

```

```
python l0_vat_sup_inductive.py --trainer=inl0 --dataset=mnist --arch=MLPSup --data-dir=data --vis --lr=0.001 --lr-decay=0.95 --lr-a=0.000001 --epoch-decay-start=100 --num-epochs=100 --lamb=1  --alpha=1 --k=1 --layer=1 --batch-size=100 --num-batch-it=500 --eps=2 --debug --log-arg=trainer-data_dir-arch-lr-lr_a-eps-lamb-top_bn-layer-debug-seed-fig --seed=1 --gpu-id=4 --alpha=1
```



## Semi-Supervised Learning

See the bash files in scripts/

### transductive way

    python l0_vat_semi_trans.py --trainer=l0 --dataset=svhn --arch=CNN9

### inductive way

    python l0_vat_semi_inductive.py --trainer=inl0 --dataset=cifar10 --arch=CNN9



## Citation

If you found this code useful, please cite our paper.

```latex
@article{xvat2020,
	title={Learning with Multiplicative Perturbations},
	author={Xiulong Yang and Shihao Ji},
	journal={International Conference on Pattern Recognition (ICPR)},
	year={2020}
}
```

