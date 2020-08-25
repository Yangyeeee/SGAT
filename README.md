# SGAT

This reposityory contains the PyTorch implementation for [Sparse Graph Attention Networks](https://arxiv.org/abs/1912.00552)



## Demo

The evolution of the graph of Zachary's Karate Club at different training epochs:

![demo](https://github.com/sndnyang/xvat/raw/master/demo/moons.gif)

## Requirements

    PyTorch >= 0.4.0
    tensorboardX <= 1.6.0
    numpy



# Usage

## Basic Parameters

```
optional arguments:
("--gpu", type=int, default=-1,                                      
 help="which GPU to use. Set -1 to use CPU.")                        
("--epochs", type=int, default=400,                                  
 help="number of training epochs")                                   
("--num-heads", type=int, default=4,                                 
 help="number of hidden attention heads")                            
("--num-out-heads", type=int, default=6,                             
 help="number of output attention heads")                            
("--num-layers", type=int, default=2,                                
 help="number of hidden layers")                                     
("--num-hidden", type=int, default=256,                              
 help="number of hidden units")                                      
("--residual", action="store_true", default=True,                    
 help="use residual connection")                                     
("--in-drop", type=float, default=0,                                 
 help="input feature dropout")                                       
("--attn-drop", type=float, default=0,                               
 help="attention dropout")                                           
("--lr", type=float, default=0.005,                                  
 help="learning rate")                                               
('--weight-decay', type=float, default=0,                            
 help="weight decay")                                                
('--alpha', type=float, default=0.2,                                 
 help="the negative slop of leaky relu")                             
('--batch-size', type=int, default=2,                                
 help="batch size used for training, validation and test")           
('--patience', type=int, default=10,                                 
 help="used for early stop")                                         
('--seed', type=int, default=123, help='Random seed.')               
('--loss_l0', type=float, default=0, help=loss for L0 regularization.')  
("--l0", type=int, default=0, help="l0")                             
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


## Citation

If you found this code useful, please cite our paper.

```latex
@article{sgat2020,
	title={Sparse Graph Attention Networks},
	author={Yang Ye and Shihao Ji},
	journal={arXiv preprint arXiv:1912.00552 },
	year={2019}
}
```

