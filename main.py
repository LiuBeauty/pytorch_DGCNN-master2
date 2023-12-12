
import sys
from functools import reduce
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
print('On DGCNN_embeding start')
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier
from sklearn import metrics
from util import net_paramter, load_data, load_data_for_readsClassification
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import csv
where = []


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

def find_elements_with_frequency_greater_than_half(lst,fre):
    frequency_count = {}
    result = []

    # 计算每个元素的出现次数
    for element in lst:
        if element in frequency_count:
            frequency_count[element] += 1
        else:
            frequency_count[element] = 1

    # 筛选出出现次数大于 10 的元素
    for element, frequency in frequency_count.items():
        if frequency > fre:
            result.append(element)

    return result

def loop_dataset(g_list, classifier, sample_idxes, optimizer, bsize=net_paramter.batch_size):
    total_loss = []
    pbar = tqdm(range(len(sample_idxes)), unit='batch')
    all_targets = []
    all_scores = []
    n_samples = 0
    all_logits = []
    imp_genes = []

    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        npos_graph = len(selected_idx)
        logits, loss, acc,top_indices = classifier(batch_graph)

        all_scores.append(logits[:, 1].cpu().detach())
        logits2 = logits.cpu().detach().numpy()
        num_class = logits2.shape[1]
        all_logits = np.append(all_logits, logits2)
        # for binary classification
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            imp_genes.append(top_indices[:20])


        loss = loss.data.cpu().detach().numpy()
        total_loss.append(np.array([loss, acc]) * len(selected_idx))
        n_samples += len(selected_idx)

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_logits = all_logits.reshape(-1, num_class)

    total_gene = []
    for each in imp_genes:
        for each_gene in each:
            total_gene.append(each_gene)

    top_indices = find_elements_with_frequency_greater_than_half(total_gene,len(sample_idxes)/2)


    return avg_loss, all_logits ,top_indices


def get_oof(clf, n_folds, X_train, y_train, X_test):
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    classnum = len(np.unique(y_train))
    kf = KFold(n_splits=n_folds, shuffle=False)
    oof_train = np.zeros((ntrain, classnum))
    oof_test = np.zeros((ntest, classnum))

    pd.set_option('display.max_rows', None)  # 显示全部行
    pd.set_option('display.max_columns', None)  # 显示全部列

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index]  # 数据
        kf_y_train = y_train[train_index]  # 标签
        kf_X_test = X_train[test_index]  # k-fold的验证集

        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index] = clf.predict_proba(kf_X_test)
        oof_test += clf.predict_proba(X_test)

    oof_test = oof_test / float(n_folds)
    return oof_train, oof_test,clf


def get_off_dgcnn(clf, n_folds, graph_train, graph_test, classnum):
    ntrain = len(graph_train)

    ntest = len(graph_test)
    kf = KFold(n_splits=n_folds, shuffle=False)
    oof_train = np.zeros((ntrain, classnum))
    oof_test = np.zeros((ntest, classnum))
    optimizer = optim.Adam(clf.parameters(), lr=net_paramter.learning_rate)
    test_idxes = list(range(len(graph_test)))
    top_indices = []

    for i, (train_index, test_index) in enumerate(kf.split(graph_train)):
        kf_X_train = [graph_train[j] for j in train_index]  # 数据
        kf_X_test = [graph_train[j] for j in test_index]  # k-fold的验证集
        X_train_idxes = list(range(len(kf_X_train)))
        X_test_idxes = list(range(len(kf_X_test)))

        clf.train()
        for epoch in range(net_paramter.num_epochs):
            avg_loss, _ ,_= loop_dataset(kf_X_train,clf, X_train_idxes, optimizer=optimizer)
        clf.eval()

        _, oof_train[test_index],_ = loop_dataset(kf_X_test, clf, X_test_idxes, optimizer=None)
        _, off_test_demo ,top_index= loop_dataset(graph_test, clf, test_idxes, optimizer=None)
        oof_test += off_test_demo
        top_indices.append(top_index)


    oof_test = oof_test / float(n_folds)


    return oof_train, oof_test,top_indices


def SelectModel(modelname):
    if modelname == "SVM":
        from sklearn.svm import SVC
        c = pow(2, -10)
        model = SVC(C=c, kernel='linear', probability=True, random_state=net_paramter.seed)

    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()

    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

    elif modelname == "XGBOOST":
        import xgboost as xgb
        model = xgb.XGBClassifier(seed = net_paramter.seed)

    elif modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        model = knn()
    else:
        pass
    return model


def tradion_load_data():
    data_train = list()
    data_test = list()


    total_expression_path = 'data/BRCA/BRCA_gene_expression.csv'
    total_expression_feature = pd.read_csv(total_expression_path, header=0,index_col=0)

    total_bodyMeth_path = 'data/BRCA/BRCA_bodyMeth.csv'
    total_bodyMeth_feature = pd.read_csv(total_bodyMeth_path, header=0,index_col=0)

    total_proMeth_path = 'data/BRCA/BRCA_proMeth.csv'
    total_proMeth_feature = pd.read_csv(total_proMeth_path, header=0,index_col=0)

    col_names = list(total_proMeth_feature.columns)
    col_names.remove('subtype')
    ncol = total_proMeth_feature.shape[1] - 1

    subtypedic = {'Basal':0,'Normal':1,'Her2':2,'LumA':3,'LumB':4}

    label = (total_expression_feature.iloc[:, -1]).to_numpy()
    label = [subtypedic[item] for item in label]
    label = np.array(label)

    proMeth_feature = (total_proMeth_feature.iloc[:, 0:ncol]).to_numpy()
    bodyMeth_feature = (total_bodyMeth_feature.iloc[:, 0:ncol]).to_numpy()
    expression_feature = (total_expression_feature.iloc[:, 0:ncol]).to_numpy()

    indice = np.arange(0, total_expression_feature.shape[0] - 1)
    data_train_index, data_test_index, label_train_index, label_test_index = train_test_split(indice, indice,
                                                                                              random_state=0,
                                                                                              test_size=0.2)

    data_train.append(expression_feature[data_train_index])
    label_train = label[data_train_index]
    data_test.append(expression_feature[data_test_index])
    label_test = label[data_test_index]

    data_train.append(proMeth_feature[data_train_index])
    data_test.append(proMeth_feature[data_test_index])

    data_train.append(bodyMeth_feature[data_train_index])
    data_test.append(bodyMeth_feature[data_test_index])

    col_names = list(total_proMeth_feature.columns)
    col_names.remove('subtype')
    net_paramter.feat_name = col_names

    return data_train, data_test, label_train, label_test, data_train_index, data_test_index,col_names

def top_n(lst, n):
    """
    返回前 N 大的元素及其在列表中的位置
    """
    # 将列表转化为 numpy 数组

    arr = np.array(lst)
    # 获取元素在排序之后的位置
    sorted_indexes = np.argsort(arr)[::-1][:n]
    # 获取元素和其位置
    #top_n_values = [(lst[i], i) for i in sorted_indexes]
    top_values = [lst[i] for i in sorted_indexes]
    top_indices= [i for i in sorted_indexes]

    return top_indices,top_values


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONSHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True


if __name__ == '__main__':
    print('main start')
    modelist = ['SVM', 'SVM', 'SVM', 'XGBOOST', 'XGBOOST']

    setup_seed(net_paramter.seed)

    data_train, data_test, label_train, label_test, data_train_index, data_test_index,col_names = tradion_load_data()
    print(f'---{len(col_names)}')
    num_class = len(np.unique(label_train))
    graphs = load_data()
    train_graphs = [graphs[i] for i in data_train_index]
    test_graphs = [graphs[i] for i in data_test_index]

    # if net_paramter.sortpooling_k <= 1:
    #     argst = np.argsort([g.num_nodes for g in graphs])
    #     num_nodes_list = sorted([g.num_nodes for g in graphs])
    #     net_paramter.sortpooling_k = num_nodes_list[int(math.ceil(net_paramter.sortpooling_k * len(num_nodes_list))) - 1]
    #     net_paramter.sortpooling_k = max(10, net_paramter.sortpooling_k)
    #     print('k used in SortPooling is: ' + str(net_paramter.sortpooling_k))

    classifier = Classifier()
    if net_paramter.mode == 'gpu':
        classifier = classifier.cuda()

    newfeature_list = []
    newtestdata_list = []

    print('--------------------First1.1 layer calculating-------DGCNN')
    oof_train_, oof_test_,top_indices = get_off_dgcnn(clf=classifier, n_folds=5, graph_train=train_graphs, graph_test=test_graphs,
                                          classnum=num_class)
    # newfeature_list.append(oof_train_)
    # newtestdata_list.append(oof_test_)

    dgcnn_imp_gene = [item for sublist in top_indices for item in sublist]
    dgcnn_imp_gene = [col_names[i] for i in dgcnn_imp_gene]
    imp_gene_dgcnn = [[x] for x in dgcnn_imp_gene]
    filename = './result/important_gene_by_dgcnn.csv'

    # with open(filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for row in imp_gene_dgcnn:
    #         writer.writerow(row)


    newfeature_list2 = []
    newtestdata_list2 = []
    omics_dic = {0: 'Feature importance for expression',
                 1: 'Feature importance for Promoter methylation',
                 2: 'Feature importance for Body methylation'}

    for index in range(0, 3):
        title = omics_dic[index]
        clf_first = SelectModel(modelist[index])
        oof_train_, oof_test_,clf_trained = get_oof(clf=clf_first, n_folds=5, X_train=data_train[index],
                                        y_train=label_train,
                                        X_test=data_test[index])
        coef = clf_trained.coef_
        to_sort = np.abs(coef[0])
        print(f'-----to_sort {len(to_sort)}')
        top_indices, top_values = top_n(to_sort, 20)
        gene = [col_names[i] for i in top_indices]

        plt.figure(figsize=(16, 12))  # 调整图形的大小
        plt.bar(range(20), top_values)
        plt.xticks(range(20), gene, rotation=90)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title(title)
        plt.savefig(f'./feature_importance/{title}.png')  # 保存图形
        plt.clf()  # 清空当前的图形，以便绘制下一个图形

        top_indices_for_analysis, _ = top_n(to_sort,200)
        gene_for_analysis = [col_names[i] for i in top_indices_for_analysis]
        imp_gene = [[x] for x in gene_for_analysis ]

        filename = f'./result/important_gene_by_svm_{title}.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in imp_gene:
                writer.writerow(row)

        newfeature_list2.append(oof_train_)
        newtestdata_list2.append(oof_test_)


    newtraindata1 = reduce(lambda x, y: np.concatenate((x, y), axis=1), newfeature_list2)
    newtestdata1 = reduce(lambda x, y: np.concatenate((x, y), axis=1), newtestdata_list2)


    print('-------Second layer calculating--------')
    clf_second = SelectModel(modelist[3])
    oof_train_, oof_test_ ,clf_trained= get_oof(clf=clf_second, n_folds=5, X_train=newtraindata1,
                                    y_train=label_train,
                                    X_test=newtestdata1 )

    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 12))
    xgb.plot_importance(clf_trained, ax=ax, importance_type='weight', max_num_features=20)
    #plt.savefig('./feature_importance/xgboost_feature_importance_svm_stacking1.png')

    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)

    newtraindata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newfeature_list)
    newtestdata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newtestdata_list)

    print('--------------Third layer calculating-------')
    clf_second1 = SelectModel(modelist[4])
    clf_second1.fit(newtraindata, label_train)
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(clf_second1, ax=ax, importance_type='weight', max_num_features=20)
    #plt.savefig('./feature_importance/xgboost_feature_importance_stacking2.png')
    pred = clf_second1.predict(newtestdata)

    cancer_type = net_paramter.data
    if num_class == 2:
        F1_score = f1_score(label_test, pred)
        accuracy = accuracy_score(label_test, pred)
        score = roc_auc_score(label_test, pred)
        print(f'{cancer_type}  accuracy:  {accuracy}\n  F1_macro: {F1_score} \n roc_auc_score: {score}')

    else:
        F1_score = f1_score(label_test, pred, average='macro')
        F1_weight = f1_score(label_test, pred, average='weighted')
        right_number = 0
        for i in range(0, len(pred) - 1):
            if pred[i] == label_test[i]:
                right_number += 1
        accuracy = right_number / len(label_test)
        print(f'{cancer_type} :  accuracy:  {accuracy}\n  F1_macro: {F1_score} \n F1_weight: {F1_weight}  right_number{right_number}  len_lebel_test:{len(label_test)} ')

    sns.set()
    fig, ax = plt.subplots()
    # for i in range(len(pred)):
    #     if pred[i] == 0:
    #         pred[i] = 1
    # for i in range(len(label_test)):
    #     if label_test[i] == 0:
    #         label_test[i] = 1

    # print(label_test)
    # print(pred)

    C2 = confusion_matrix(label_test, pred, labels=[0, 1, 2, 3, 4])
    im = ax.imshow(C2,cmap = plt.cm.Blues)
    ax.set_xticks([0,1,2,3,4])
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticklabels([0,1, 2, 3, 4])
    for i in range(len([0,1,2,3,4])):
        for j in range(len([0,1,2,3,4])):
            text = ax.text(j,i,C2[i,j],ha = 'center',va = 'center',color = 'white' if C2[i,j] > (C2.max() / 2) else 'black')

    ax.set_title('BRCA confusion matrix with graph stacking')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.colorbar(im)
    plt.savefig('result/BRCA_result1.png')

    # plt.figure(1)
    # plt.plot(num_epoch, train_loss1,'r',label = 'Train loss')
    # plt.plot(num_epoch, test_loss1, 'b', label='Test loss')
    # plt.legend()
    # plt.savefig('/public/jxliu/testProject2/RESULT/Reads_classification/loss_BRCA.png')
    #
    # plt.figure(2)
    # plt.plot(num_epoch, train_acc1,'r',label = 'Train Accuracy')
    # plt.plot(num_epoch, test_acc1, 'b', label='Test Accruay')
    # plt.legend()
    # plt.axhline(0.7)
    # plt.axhline(0.8)
    # plt.savefig('/public/jxliu/testProject2/RESULT/Reads_classification/accuracy_BRCA.png')


