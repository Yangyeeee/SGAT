import argparse
import time
from time import localtime
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gat import GAT
from torch.utils.tensorboard import SummaryWriter
import random
from torch.backends import cudnn
from reddit import RedditDataset
from ms import MsDataset
import networkx as nx
import numpy as np
from utils import EarlyStopping


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, mask,loss_fcn):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        loss_data = loss_fcn(logits, labels)
        return accuracy(logits, labels), loss_data

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

def main(args):
    # load and preprocess dataset
    if args.dataset == 'reddit':
        data = RedditDataset()
    elif args.dataset in ['photo', "computer"]:
        data = MsDataset(args)
    else:
        data = load_data(args)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    current_time = time.strftime('%d_%H:%M:%S', localtime())
    writer = SummaryWriter(log_dir='runs/' + current_time + '_' + args.sess, flush_secs=30)

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.bool().cuda()
        val_mask = val_mask.bool().cuda()
        test_mask = test_mask.bool().cuda()


    g = data.graph
    # add self loop
    if args.dataset != 'reddit':
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()
    print('edge number %d'%(n_edges))
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.idrop,
                args.adrop,
                args.alpha,
                args.residual, args.l0)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=150)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dur = []
    time_used = 0

    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        loss_l0 = args.loss_l0*( model.gat_layers[0].loss)
        optimizer.zero_grad()
        (loss + loss_l0).backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        writer.add_scalar('edge_num/0', model.gat_layers[0].num, epoch)

        if args.fastmode:
            val_acc, loss = accuracy(logits[val_mask], labels[val_mask], loss_fcn)
        else:
            val_acc,_ = evaluate(model, features, labels, val_mask, loss_fcn)
            if args.early_stop:
                if stopper.step(val_acc, model):   
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))
        writer.add_scalar('loss', loss.item(), epoch)
        writer.add_scalar('f1/train_f1_mic', train_acc, epoch)
        writer.add_scalar('f1/test_f1_mic', val_acc, epoch)
        writer.add_scalar('time/time', time_used, epoch)

    writer.close()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    acc, _ = evaluate(model,features, labels, test_mask, loss_fcn)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--l0", type=int, default=0, help="l0")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--idrop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--adrop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--loss_l0', type=float, default=0, help='loss for L0 regularization')
    parser.add_argument("--syn_type", type=str, default='scipy', help="reddit")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument('--sess', default='default', type=str, help='session id')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    set_seeds(args.seed)
    main(args)
