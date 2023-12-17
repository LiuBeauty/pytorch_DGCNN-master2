import sys
import os
import torch
import random
import numpy as np
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
        self.gnn = DGCNN(latent_dim=net_paramter.latent_dim,
                         output_dim=net_paramter.out_dim,
                         num_node_feats=net_paramter.feat_dim + net_paramter.attr_dim,
                         num_edge_feats=net_paramter.edge_feat_dim,
                         k=net_paramter.sortpooling_k,
                         conv1d_activation=net_paramter.conv1d_activation)

        out_dim = self.gnn.dense_dim


        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=net_paramter.hidden, num_class=net_paramter.num_class,
                                 with_dropout=net_paramter.dropout)



    def forward(self, batch_graph):
        with torch.no_grad():  # Make tensor's  requires_grad = False
            feature_label = self.PrepareFeatureLabel(batch_graph)
            node_feat, labels = feature_label
            edge_feat = None

        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels)



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



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)


    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions

    if net_paramter.printAUC:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    else:
        avg_loss = np.concatenate((avg_loss, [0.0]))

    return avg_loss


if __name__ == '__main__':
    print(net_paramter)
    random.seed(net_paramter.seed)
    np.random.seed(net_paramter.seed)
    torch.manual_seed(net_paramter.seed)

    train_graphs, test_graphs = load_data()
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
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not net_paramter.printAUC:
            avg_loss[2] = 0.0
        train_loss1.append(avg_loss[0])
        train_acc1.append(avg_loss[1])
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        epoch, avg_loss[0], avg_loss[1], avg_loss[2]))
        classifier.eval()


        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))

        test_loss1.append(test_loss[0])
        test_acc1.append(test_loss[1])
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))



    plt.figure(1)
    plt.plot(num_epoch, train_loss1, 'r', label='Train loss')
    plt.plot(num_epoch, test_loss1, 'b', label='Test loss')
    plt.legend()
    plt.savefig('/public/jxliu/testProject2/RESULT/Reads_classification/loss_BRCA.png')

    plt.figure(2)
    plt.plot(num_epoch, train_acc1, 'r', label='Train Accuracy')
    plt.plot(num_epoch, test_acc1, 'b', label='Test Accruay')
    plt.legend()

    plt.savefig('/public/jxliu/testProject2/RESULT/Reads_classification/accuracy_BRCA.png')

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
