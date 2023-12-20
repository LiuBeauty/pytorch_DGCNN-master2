import sys
import os
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier
from sklearn import metrics
from util import net_paramter, load_data
import matplotlib.pyplot as plt
import DGCNN_embedding as de
from time import sleep
import csv



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.gnn = DGCNN(epoch = net_paramter.num_epochs,latent_dim=net_paramter.latent_dim,
                         num_node_feats=net_paramter.feat_dim + net_paramter.attr_dim,
                         k=net_paramter.sortpooling_k,
                         conv1d_channels= net_paramter.conv1d_channels,
                         conv1d_kws=net_paramter.conv1d_kws,
                         conv1d_activation=net_paramter.conv1d_activation)

        out_dim = self.gnn.dense_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=net_paramter.hidden, num_class=net_paramter.num_class,
                                 with_dropout=net_paramter.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        concat_tag = []
        concat_feat = []

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            concat_tag += batch_graph[i].node_tags
            tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
            concat_feat.append(tmp)

        concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
        node_tag = torch.zeros(n_nodes, net_paramter.feat_dim)
        node_tag.scatter_(1, concat_tag, 1)

        node_feat = torch.cat(concat_feat, 0)
        node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)

        if net_paramter.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):

        feature_label = self.PrepareFeatureLabel(batch_graph)
        node_feat, labels = feature_label
        embed,top_indices = self.gnn(batch_graph,node_feat)

        return self.mlp(top_indices,embed, labels)




def creat_csv(path):
    with open(path, 'wb') as f:
        csv_write = csv.writer(f)
        print('------------grad_file generating--------')


def _write_msg(path, msg):
    if not os.path.exists(path):
        creat_csv(path)
    with open(path, mode='a+', encoding='utf-8', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerows(msg)


def _write_tensor(path, msg):
    if not os.path.exists(path):
        creat_csv(path)
    with open(path, mode='a+', encoding='utf-8', newline='') as f:
        msg = msg.cpu().numpy()
        csv_write = csv.writer(f)
        for each in msg:
            csv_write.writerow(msg[each])
        print(msg)


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=net_paramter.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []
    n_samples = 0

    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc, y = classifier(batch_graph)
        all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        #pbar.set_description('epoch:%d loss: %0.5f acc: %0.5f' % (loss, acc,epoch))
        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)


    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    avg_loss = np.concatenate((avg_loss, [0.0]))

    return avg_loss


if __name__ == '__main__':
    print(net_paramter)
    random.seed(net_paramter.seed)
    np.random.seed(net_paramter.seed)
    torch.manual_seed(net_paramter.seed)

    graphs = load_data()
    indice = np.arange(0, len(graphs) - 1)

    data_train_index, data_test_index, label_train_index, label_test_index = train_test_split(indice, indice,
                                                                                              random_state=0,
                                                                                              test_size=0.2)

    train_graphs = [graphs[i] for i in data_train_index]
    test_graphs = [graphs[i] for i in data_test_index]

    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if net_paramter.sortpooling_k <= 1:
        argst = np.argsort([g.num_nodes for g in train_graphs + test_graphs])
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        net_paramter.sortpooling_k = num_nodes_list[int(math.ceil(net_paramter.sortpooling_k * len(num_nodes_list))) - 1]
        net_paramter.sortpooling_k = max(10, net_paramter.sortpooling_k)
        print('k used in SortPooling is: ' + str(net_paramter.sortpooling_k))

    classifier = Classifier()
    if net_paramter.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=net_paramter.learning_rate)

    train_idxes = list(range(len(train_graphs)))

    best_loss = None
    train_acc1 = []
    train_loss1 = []
    test_acc1 = []
    test_loss1 = []
    num_epoch = []
    # net_paramter.num_epochs
    for epoch in range(net_paramter.num_epochs):
        num_epoch.append(epoch)

        classifier.train()

        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)

        train_loss1.append(avg_loss[0])
        train_acc1.append(avg_loss[1])
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f \033[0m' % (epoch, avg_loss[0], avg_loss[1]))

        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        test_loss1.append(test_loss[0])
        test_acc1.append(test_loss[1])
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f \033[0m' % (epoch, test_loss[0], test_loss[1]))

    plt.figure(1)
    plt.plot(num_epoch, train_loss1, 'r', label='Train loss')
    plt.plot(num_epoch, test_loss1, 'b', label='Test loss')
    plt.legend()
    plt.savefig('/data/home/jxliu/pytorch_DGCNN-master2/result/loss_BRCA_only_DGCNN.png')

    plt.figure(2)
    plt.plot(num_epoch, train_acc1, 'r', label='Train Accuracy')
    plt.plot(num_epoch, test_acc1, 'b', label='Test Accruay')
    plt.legend()
    plt.savefig('/data/home/jxliu/pytorch_DGCNN-master2/result/accuracy_BRCA_only_DGCNN.png')

"""
    # plt.figure(3)
    # plt.plot(num_epoch,test_loss1,'b',label = 'Test loss')
    # plt.legend()
    # plt.savefig('/public/jxliu/testProject2/RESULT/TESTLIU5_new_test_loss.png')
    #
    # plt.figure(4)
    # plt.plot(num_epoch,test_acc1,'r',label = 'Test acc')
    # plt.legend()
    # plt.savefig('/public/jxliu/testProject2/RESULT/TESTLIU5_new_test_acc.png')
    '''
    with open(net_paramter.data + '_acc_results.txt', 'a+') as f:
        f.write(str(test_loss[1]) + '\n')

    if net_paramter.printAUC:
        with open(net_paramter.data + '_auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    if net_paramter.extract_features:
        features, labels = classifier.output_features(train_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_train.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
        features, labels = classifier.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_test.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
"""
