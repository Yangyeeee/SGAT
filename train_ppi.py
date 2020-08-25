

import numpy as np
import torch
import dgl
import time
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score
from gat import GAT
from dgl.data.ppi import LegacyPPIDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
from torch.backends import cudnn

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

def collate(sample):
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

def evaluate(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0., 1, 0)
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        return score, loss_data.item()
        
def main(args):
    if args.gpu<0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))
    writer = SummaryWriter()
    batch_size = args.batch_size
    # cur_step = 0
    # patience = args.patience
    # best_score = -1
    # best_loss = 10000
    # define loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    # create the dataset
    train_dataset = LegacyPPIDataset(mode='train')
    valid_dataset = LegacyPPIDataset(mode='valid')
    test_dataset = LegacyPPIDataset(mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    n_classes = train_dataset.labels.shape[1]
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

    # define the model
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.residual, args.l0)
    print(model)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)
    best_epoch = 0
    dur = []
    acc = []
    for epoch in range(args.epochs):
        num = 0
        model.train()
        if epoch % 5 == 0:
            t0 = time.time()
        loss_list = []
        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            feats = feats.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(feats.float())
            loss = loss_fcn(logits, labels.float())
            loss_l0 = args.reg1 *(model.gat_layers[0].loss)
            optimizer.zero_grad()
            (loss+loss_l0).backward()
            optimizer.step()
            loss_list.append(loss.item())
            num += model.gat_layers[0].num

        if epoch % 5 == 0:
            dur.append(time.time() - t0)

        loss_data = np.array(loss_list).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
        writer.add_scalar('edge_num/0', num, epoch)

        if epoch%5 == 0:
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                feats = feats.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)

            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            print("val F1-Score: {:.4f} ".format(mean_score))
            writer.add_scalar('loss', mean_val_loss, epoch)
            writer.add_scalar('f1/test_f1_mic', mean_score, epoch)

            acc.append(mean_score)

            # # early stop
            # if mean_score > best_score or best_loss > mean_val_loss:
            #     if mean_score > best_score and best_loss > mean_val_loss:
            #         val_early_loss = mean_val_loss
            #         val_early_score = mean_score
            #         torch.save(model.state_dict(), '{}.pkl'.format('save_rand'))
            #         best_epoch = epoch
            #
            #     best_score = np.max((mean_score, best_score))
            #     best_loss = np.min((best_loss, mean_val_loss))
            #     cur_step = 0
            # else:
            #     cur_step += 1
            #     if cur_step == patience:
            #         break


    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
    acc = np.array(test_score_list).mean()
    print("test F1-Score: {:.4f}".format(acc))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--reg1', type=float, default=0, help='Alpha for the leaky_relu.')
    parser.add_argument('--reg2', type=float, default=0, help='Alpha for the leaky_relu.')
    parser.add_argument("--l0", type=int, default=0, help="l0")
    args = parser.parse_args()
    print(args)
    set_seeds(args.seed)
    main(args)