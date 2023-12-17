from __future__ import print_function
import numpy as np

import networkx as nx
import pdb
import argparse


# cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
#
# cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
# cmd_opt.add_argument('-gm', default='DGCNN', help='gnn model to use')
# cmd_opt.add_argument('-data', default= 'MUTAG', help='data folder name')
# cmd_opt.add_argument('-batch_size', type=int, default=100, help='minibatch size')
# cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
# cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
# cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
# cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
# cmd_opt.add_argument('-fold', type=int, default=2, help='fold (1..10)')
# cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
# cmd_opt.add_argument('-num_epochs', type=int, default=20, help='number of epochs')
# cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
# cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
# cmd_opt.add_argument('-conv1d_activation', type=str, default='Tanh', help='which nn activation layer to use')
# cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
# cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of mlp hidden layer')
# cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
# cmd_opt.add_argument('-learning_rate', type=float, default=0.01, help='init learning_rate')
# cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
# cmd_opt.add_argument('-printAUC', type=bool, default=False, help='whether to print AUC (for binary classification only)')
# cmd_opt.add_argument('-extract_features', type=bool, default=True, help='whether to extract final graph features')
#
# net_paramter, _ = cmd_opt.parse_known_args()
# net_paramter.latent_dim = [int(x) for x in net_paramter.latent_dim.split('-')]

class net_paramter:
    attr_dim = 1
    data = 'BRCA'
    batch_size = 1
    seed = 0
    feat_dim = 3
    test_number = 175
    num_epochs = 15
    latent_dim = '26-24-24-1'
    sortpooling_k = 0.5
    conv1d_activation = 'Tanh'
    out_dim = 0
    learning_rate = 0.00001
    dropout = False
    hidden = 128
    mode = 'gpu'
    conv1d_channels = [16, 32]
    conv1d_kws = [0, 5]
    feat_name = []

net_paramter.latent_dim = [int(x) for x in net_paramter.latent_dim.split('-')]

if len(net_paramter.latent_dim) == 1:
    net_paramter.latent_dim = net_paramter.latent_dim[0]

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        deg_dict = dict(g.degree)
        new_deg_dict = {}
        for i in sorted(deg_dict):
            new_deg_dict[i] = deg_dict[i]
        self.degs = list(new_deg_dict.values())

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()

        # no edge features
        self.edge_features = None

def load_data():
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('data2/%s/%s.txt' % (net_paramter.data, net_paramter.data), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                node_features.append(attr)
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])
                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            node_features = np.stack(node_features)
            node_feature_flag = True
            assert len(g) == n
            g_list.append(GNNGraph(g, l, node_tags, node_features))

    for g in g_list:
        g.label = label_dict[g.label]

    net_paramter.num_class = len(label_dict)
    net_paramter.feat_dim = len(feat_dict) #node_tag
    net_paramter.edge_feat_dim = 0
    net_paramter.attr_dim = node_features.shape[1] # dim of node features (attributes)


    print('# classes: %d' % net_paramter.num_class)
    print('# maximum node tag: %d' % net_paramter.feat_dim)

    return g_list


# def load_data_for_readsClassification():
#
#     print('loading data')
#     g_list_train = []
#     label_dict = {}
#     feat_dict = {}
#
#     with open('/public/jxliu/testProject2/pytorch_DGCNN-master/data/Reads_classification/train.txt' ) as f:
#         n_g = int(f.readline().strip())
#         for i in range(n_g):
#             row = f.readline().strip().split()
#             n, l = [int(w) for w in row]
#             if not l in label_dict:
#                 mapped = len(label_dict)
#                 label_dict[l] = mapped
#             g = nx.Graph()
#             node_tags = []
#             node_features = []
#             n_edges = 0
#             for j in range(n):
#                 g.add_node(j)
#                 row = f.readline().strip().split()
#                 tmp = int(row[1]) + 2
#                 row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
#                 node_features.append(attr)
#                 if not row[0] in feat_dict:
#                     mapped = len(feat_dict)
#                     feat_dict[row[0]] = mapped
#                 node_tags.append(feat_dict[row[0]])
#                 n_edges += row[1]
#                 for k in range(2, len(row)):
#                     g.add_edge(j, row[k])
#
#             node_features = np.stack(node_features)
#             node_feature_flag = True
#             assert len(g) == n
#             g_list_train.append(GNNGraph(g, l, node_tags, node_features))
#
#     for g in g_list_train:
#         g.label = label_dict[g.label]
#     g_list_test = []
#     with open('/public/jxliu/testProject2/pytorch_DGCNN-master/data/Reads_classification/test.txt' ) as f:
#         n_g = int(f.readline().strip())
#         for i in range(n_g):
#             row = f.readline().strip().split()
#             n, l = [int(w) for w in row]
#             if not l in label_dict:
#                 mapped = len(label_dict)
#                 label_dict[l] = mapped
#             g = nx.Graph()
#             node_tags = []
#             node_features = []
#             n_edges = 0
#             for j in range(n):
#                 g.add_node(j)
#                 row = f.readline().strip().split()
#                 tmp = int(row[1]) + 2
#                 row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
#                 node_features.append(attr)
#                 if not row[0] in feat_dict:
#                     mapped = len(feat_dict)
#                     feat_dict[row[0]] = mapped
#                 node_tags.append(feat_dict[row[0]])
#                 n_edges += row[1]
#                 for k in range(2, len(row)):
#                     g.add_edge(j, row[k])
#
#             node_features = np.stack(node_features)
#             node_feature_flag = True
#             assert len(g) == n
#             g_list_test.append(GNNGraph(g, l, node_tags, node_features))
#
#     for g in g_list_test:
#         g.label = label_dict[g.label]
#
#
#     net_paramter.num_class = len(label_dict)
#     net_paramter.feat_dim = len(feat_dict) # maximum node label (tag)
#     net_paramter.edge_feat_dim = 0
#     if node_feature_flag == True:
#         net_paramter.attr_dim = node_features.shape[1] # dim of node features (attributes)
#     else:
#         net_paramter.attr_dim = 0
#
#     print('# classes: %d' % net_paramter.num_class)
#     print('# maximum node tag: %d' % net_paramter.feat_dim)
#
#
#     return g_list_train, g_list_test
