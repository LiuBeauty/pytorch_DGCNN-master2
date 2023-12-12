from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from lib.pytorch_util import weights_init


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()
        # print(f'------input size {input_size}-------------hidden_size{hidden_size}')
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, top_indices,x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.h2_weights(h1)
        logits = F.softmax(logits, dim=1)

        y = Variable(y)
        loss = criterion(logits, y)
        pred = logits.data.max(1, keepdim=True)[1]
        acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])

        return logits, loss, acc,top_indices

