# SGAT
This reposityory contains the PyTorch implementation of "Sparse Graph Attention Networks", IEEE TKDE 2021.

## Demo
The evolution of the graph of Zachary's Karate Club at different training epochs:
<img src="https://github.com/Yangyeeee/SGAT/blob/master/demo/toy.gif" width="60%"/>
## Requirements

    PyTorch == 1.4.0
    tensorboard == 1.14.0
    dgl == 0.4.3.post2  


## Usage
### Basic Parameters
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

### Cora
```
python train.py --dataset=cora --l0=1 --lr=0.01 --num-heads=2 --gpu=4 --num-layers=1 --epochs=200 --num-hidden=64 --loss_l0 1e-6 --sess cora --idrop 0.4 --adrop 0.3
```

### Citeseer
```
python train.py --dataset=citeseer --l0=1 --lr=0.01 --num-heads=2 --gpu=4 --num-layers=1 --epochs=200 --num-hidden=32 --loss_l0 1e-6 --sess citeseer --idrop 0.4 --adrop 0.5
```

### Pubmed
```
python train.py --dataset=pubmed --l0=1 --lr=0.01 --num-heads=2 --gpu=4 --num-layers=1 --epochs=200 --num-hidden=32 --loss_l0 1e-6 --sess pubmed --idrop 0.1 --adrop 0.5 --num-out-heads=2 --weight-decay=0.001
```

### PPI
```
python train_ppi.py --l0=1 --num-heads=2 --num-hidden 512 --gpu=0 --num-layers=2 --lr=0.01 --loss_l0=7e-7
```


## Citation
If you found this code useful, please cite our paper.

    @article{SGAT2021,
      title   = {Sparse Graph Attention Networks},
      author  = {Yang Ye and Shihao Ji}, 
      journal = {IEEE Transactions on Knowledge and Data Engineering (TKDE)},
      month   = {April},
      year    = {2021}
    }
