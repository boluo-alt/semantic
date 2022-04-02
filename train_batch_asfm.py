# 每次更新版本测试
import numpy as np
import torch
import dgl
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score
from gat import GAT
from dgl.data.ppi import PPIDataset
from torch.utils.data import DataLoader
from CAD_Interface import CAD_InterFace
import time
from sklearn.metrics import classification_report


def collate(graphs):
    graph = dgl.batch(graphs)
    return graph


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


def evaluate1(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits
        labels = labels
        return accuracy(logits, labels)


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    _, indices1 = torch.max(labels, dim=1)
    correct = torch.sum(indices == indices1)
    return correct.item() * 1.0 / len(labels)


def accuracy1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def main(args):
    isTrain = 0
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    batch_size = args.batch_size
    cur_step = 0
    patience = args.patience
    best_score = -1
    best_loss = 10000

    loss_fcn = torch.nn.CrossEntropyLoss()

    train_dataset = CAD_InterFace(mode='train')
    valid_dataset = CAD_InterFace(mode='valid')
    test_dataset = CAD_InterFace(mode='test')
    exp_dataset = CAD_InterFace(mode='exp')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    exp_dataloader = DataLoader(exp_dataset, batch_size=1, collate_fn=collate)

    g = train_dataset[0]
    n_classes = train_dataset.num_labels
    n_classes_1 = train_dataset.num_labels_1
    num_feats = g.ndata['feat'].shape[1]

    g = g.int().to(device)
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
                args.residual)

    model_1 = GAT(g,
                  args.num_layers,
                  num_feats,
                  args.num_hidden,
                  n_classes_1,
                  heads,
                  F.elu,
                  args.in_drop,
                  args.attn_drop,
                  args.alpha,
                  args.residual)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)
    All_time = []
    dur = []
    time_all = time.time()
    for epoch in range(args.epochs):
        model.train()
        model_1.train()
        # if epoch >= 3:
        t0 = time.time()  # 返回当前时间的时间戳
        loss_list = []
        train_acc_list = []
        train_acc_1_list = []
        for batch, subgraph in enumerate(train_dataloader):  # print(batch, subgraph)返回的是元素及对应的索引
            subgraph = subgraph.to(device)
            model.g = subgraph
            model_1.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph

            logits = model(subgraph.ndata['feat'].float())
            loss = loss_fcn(logits, subgraph.ndata['label'])

            logits = model(subgraph.ndata['feat'].float())
            logits_1 = model_1(subgraph.ndata['feat'].float())
            loss = loss_fcn(logits, subgraph.ndata['label']) + loss_fcn(logits_1, subgraph.ndata['label_1'])
            train_acc = accuracy1(logits, subgraph.ndata['label'])
            train_acc_1 = accuracy1(logits_1, subgraph.ndata['label_1'])
            train_acc_all = 0.7 * train_acc + 0.3 * train_acc_1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if epoch >= 3:
            dur.append(time.time() - t0)
            loss_list.append(loss.item())
            train_acc_list.append(train_acc.item())
            train_acc_1_list.append(train_acc_1.item())
        loss_data = np.array(loss_list).mean()
        train_acc = np.array(train_acc_list).mean()
        train_acc_1 = np.array(train_acc_1_list).mean()
        print("Epoch {:05d} | Time(s) {:.4f} | Loss: {:.4f}| Tran_acc: {:.4f}| Tran_acc_1: {:.4f}".format(epoch + 1,
                                                                                                          np.mean(dur),
                                                                                                          loss_data,
                                                                                                          train_acc,
                                                                                                          train_acc_1))
        if epoch % 13 == 0:
            score_list = []
            val_loss_list = []
            val_loss_1_list = []
            for batch, subgraph in enumerate(valid_dataloader):
                with torch.no_grad():
                    model.eval()
                    model_1.eval()
                    subgraph = subgraph.to(device)

                    model.g = subgraph
                    model_1.g = subgraph

                    logits_val = model(subgraph.ndata['feat'].float())
                    logits_val_1 = model_1(subgraph.ndata['feat'].float())
                    val_loss = accuracy1(logits_val, subgraph.ndata['label'])
                    val_loss_1 = accuracy1(logits_val_1, subgraph.ndata['label_1'])

                    val_loss_list.append(val_loss.item())
                    val_loss_1_list.append(val_loss_1)
            mean_val_loss = np.array(val_loss_list).mean()
            print("Val_acc: {:.4f} ".format(mean_val_loss))
            # early stop
            if mean_val_loss > best_score or best_loss > mean_val_loss:
                if mean_val_loss > best_score and best_loss > mean_val_loss:
                    val_early_loss = mean_val_loss
                    val_early_score = mean_val_loss
                best_score = np.max((mean_val_loss, best_score))
                best_loss = np.min((best_loss, mean_val_loss))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break
    # 测试集
    test_loss_list = []
    test_loss_1_list = []
    for batch, subgraph in enumerate(test_dataloader):
        subgraph = subgraph.to(device)
        model.g = subgraph
        model_1.g = subgraph
        logits_test = model(subgraph.ndata['feat'].float())
        logits_test_1 = model_1(subgraph.ndata['feat'].float())
        test_loss = accuracy1(logits_test, subgraph.ndata['label'])
        test_loss_1 = accuracy1(logits_test_1, subgraph.ndata['label_1'])
        test_loss_list.append(test_loss.item())
        test_loss_1_list.append(test_loss_1)
    All_time.append(time.time() - time_all)
    print("Test Score: {:.4f}| Time(s) {:.4f} ".format(np.array(test_loss_list).mean(), np.mean(All_time)))


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
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=100,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=20,
                        help="used for early stop")
    parser.add_argument('--dropout', type=float, default=0.6,
                        help="used for dropout")
    args = parser.parse_args()
    print(args)

    main(args)
