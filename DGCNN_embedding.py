from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn


sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from lib.gnn_lib import GNNLIB
from lib.pytorch_util import weights_init, gnn_spmm

class DGCNN(nn.Module):
    def __init__(self, epoch,num_node_feats,  latent_dim, k, conv1d_channels, conv1d_kws, conv1d_activation):
        super(DGCNN, self).__init__()

        self.latent_dim = latent_dim
        self.num_node_feats = num_node_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim
        self.full_epoch = epoch
        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))
        weights_init(self)

    def sortpooling_embedding(self, node_feat, n2n_sp, subg_sp, graph_sizes, node_degs):

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat

        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        ''' sortpooling layer '''

        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)

        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0

        for i in range(subg_sp.size()[0]):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)

            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)

            # if k < self.k:
            #     to_pad = torch.zeros(self.k-k, self.total_latent_dim)
            #     if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            #         to_pad = to_pad.cuda()
            #
            #     to_pad = Variable(to_pad)
            #     sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)
        to_dense = conv1d_res.view(len(graph_sizes), -1)
        acted_conv1d = self.conv1d_activation(to_dense)
        topk_indices = topk_indices.tolist()
        return acted_conv1d,topk_indices


    def forward(self, graph_list, node_feat):

        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)


        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()

        n2n_sp = Variable(n2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        top_indices,h = self.sortpooling_embedding(node_feat, n2n_sp, subg_sp, graph_sizes, node_degs)
        return top_indices,h



