# SGAT

This reposityory contains the PyTorch implementation for Sparse Graph Attention Networks.



## Demo

The evolution of the graph of Zachary's Karate Club at different training epochs:

<img src="https://github.com/Yangyeeee/SGAT/blob/master/demo/toy.gif" width="60%"/>

## Requirements

    PyTorch == 1.4.0
    tensorboard == 1.14.0
    dgl == 0.4.3.post2  



# Usage

## Basic Parameters

```
optional arguments:
("--gpu",            type=int,   default=-1,   help="which GPU to use. Set -1 to use CPU.")                        
("--epochs",         type=int,   default=400,  help="number of training epochs")                                   
("--num-heads",      type=int,   default=4,    help="number of hidden attention heads")                            
("--num-out-heads",  type=int,   default=6,    help="number of output attention heads")                            
("--num-layers",     type=int,   default=2,    help="number of hidden layers")                                     
("--num-hidden",     type=int,   default=256,  help="number of hidden units")                                                                        
("--lr",             type=float, default=0.01, help="learning rate")                                               
('--weight-decay',   type=float, default=0,    help="weight decay")                                                       
('--loss_l0',        type=float, default=0,    help=loss for L0 regularization.')  
("--l0",             type=int,  default=0,     help="l0 regularization")                             
```


## Cora
```
python train.py --dataset=cora --l0=1 --lr=0.01 --num-heads=2 --gpu=4 --num-layers=1 --epochs=200 --num-hidden=32 --loss_l0 1e-6 --sess cora

```
## Citeseer
```
python train.py --dataset=citeseer --l0=1 --lr=0.01 --num-heads=2 --gpu=4 --num-layers=1 --epochs=200 --num-hidden=32 --loss_l0 1e-6 --sess citeseer
```
## Pubmed
```
python train.py --dataset=pubmed --l0=1 --lr=0.01 --num-heads=2 --gpu=4 --num-layers=1 --epochs=200 --num-hidden=32 --loss_l0 1e-6 --sess pubmed
```
## PPI
```
python train_ppi.py --l0=1 --num-heads=2 --gpu=0 --num-layers=2 --lr=0.01 --loss_l0=7e-7
```
