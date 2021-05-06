#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, weight_init=None, bias_init=None):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_out)

        if weight_init == 'xavier':
            torch.nn.init.xavier_uniform_(self.layer_input.weight)
            torch.nn.init.xavier_uniform_(self.layer_hidden1.weight)
            torch.nn.init.xavier_uniform_(self.layer_hidden2.weight)
        elif weight_init == 'kaiming':
            torch.nn.init.kaiming_normal_(self.layer_input.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.layer_hidden1.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.layer_hidden2.weight, mode='fan_in', nonlinearity='relu')            

        if bias_init == 'zeros':
            torch.nn.init.zeros_(self.layer_input.bias)
            torch.nn.init.zeros_(self.layer_hidden1.bias)
            torch.nn.init.zeros_(self.layer_hidden2.bias)


    def forward(self, x):
        # a and g are as defined in Algo1 of the Martens / Goose paper.  a and s would be a and h in the Goldfarb paper.
        # Notice that a and s indices are staggered.  List a would have indices starting from 0 and ends at L-1.
        # List s would start from layer index 1 and ends at L.
        a, s = [],[]

        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        a.append(x)
        x = self.layer_input(x)
        x.retain_grad()
        s.append(x)
        x = self.relu(x)

        a.append(x)
        x = self.layer_hidden1(x)
        x.retain_grad()
        s.append(x)
        x = self.relu(x)

        a.append(x)
        x = self.layer_hidden2(x)
        x.retain_grad()
        s.append(x)

        return x, s, a

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        # a and s are as defined in Algo1 of the Martens / Goose paper.  a and s would be a and h in the Goldfarb paper.
        # Notice that a and s indices are staggered.  List a would have indices starting from 0 and ends at L-1.
        # List s would start from layer index 1 and ends at L.
        a, s = [],[]

        a.append(None)
        s.append(None)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        a.append(None)
        s.append(None)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, x.shape[-3]*x.shape[2]*x.shape[3])
        a.append(x)
        x = self.fc1(x)
        x.retain_grad()
        s.append(x)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        a.append(x)
        x = self.fc2(x)
        x.retain_grad()
        s.append(x)
        #return F.log_softmax(x, dim=1)

        return x, s, a


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn0 = nn.BatchNorm1d(16 * 5 * 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        # a and s are as defined in Algo1 of the Martens / Goose paper.  a and s would be a and h in the Goldfarb paper.
        # Notice that a and s indices are staggered.  List a would have indices starting from 0 and ends at L-1.
        # List s would start from layer index 1 and ends at L.
        a, s = [],[]

        a.append(None)
        s.append(None)
        x = self.pool(F.relu(self.conv1(x)))

        a.append(None)
        s.append(None)
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = self.bn0(x)
        a.append(x)
        x = self.fc1(x)
        x.retain_grad()
        s.append(x)
        x = F.relu(x)

        x = self.bn1(x)
        a.append(x)
        x = self.fc2(x)
        x.retain_grad()
        s.append(x)
        x = F.relu(x)

        x = self.bn2(x)
        a.append(x)
        x = self.fc3(x)
        x.retain_grad()
        s.append(x)
        #return F.log_softmax(x, dim=1)
        return x, s, a

class LeNet5(nn.Module):

    def __init__(self, args):
        super(LeNet5, self).__init__()

        self.input_size = dict()
        self.output_size = dict()
        self.unfold_func = dict()

        self.conv1_kwargs = dict(kernel_size=(5, 5))
        self.conv1 = nn.Conv2d(args.num_channels, 6, **self.conv1_kwargs)
        self.unfold_func['conv1'] = nn.Unfold(**self.conv1_kwargs)
        self.input_size['conv1'] = (32, 32)
        self.output_size['conv1'] = (28, 28, 75) # 75 = kernel_size[0]*kernel_size[1]*num_channels
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2_kwargs = dict(kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, **self.conv2_kwargs)
        self.unfold_func['conv2'] = nn.Unfold(**self.conv2_kwargs)
        self.input_size['conv2'] = (14, 14)
        self.output_size['conv2'] = (10, 10, 150) # 150 = kernel_size[0]*kernel_size[1]*6
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3_kwargs = dict(kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, **self.conv3_kwargs)
        self.unfold_func['conv3'] = nn.Unfold(**self.conv3_kwargs)
        self.input_size['conv3'] = (5, 5)
        self.output_size['conv3'] = (1, 1, 400) # 400 = kernel_size[0]*kernel_size[1]*16
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, args.num_classes)
        if args.lenet5_activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            self.activation3 = nn.ReLU()
            self.activation4 = nn.ReLU()
        elif True or args.lenet5_activation == 'tanh':
            self.activation1 = nn.Tanh()
            self.activation2 = nn.Tanh()
            self.activation3 = nn.Tanh()
            self.activation4 = nn.Tanh() 

    def forward(self, img, out_feature=False):
        # a and s are as defined in Algo1 of the Martens / Goose paper.  a and s would be a and h in the Goldfarb paper.
        # Notice that a and s indices are staggered.  List a would have indices starting from 0 and ends at L-1.
        # List s would start from layer index 1 and ends at L.
        a, s = [],[]

        a_unfolded = self.unfold_func['conv1'](img)
        a.append(a_unfolded)
        # a.append(img)
        output = self.conv1(img)
        output.retain_grad()
        s.append(output)
        output = self.activation1(output)
        output = self.maxpool1(output)

        a_unfolded = self.unfold_func['conv2'](output)
        a.append(a_unfolded)
        # a.append(output)
        output = self.conv2(output)
        output.retain_grad()
        s.append(output)
        output = self.activation2(output)
        output = self.maxpool2(output)

        a_unfolded = self.unfold_func['conv3'](output)
        a.append(a_unfolded)
        # a.append(output)
        output = self.conv3(output)
        output.retain_grad()
        s.append(output)
        output = self.activation3(output)

        feature = output.view(-1, 120)
        a.append(feature)
        output = self.fc1(feature)
        output.retain_grad()
        s.append(output)
        output = self.activation4(output)

        a.append(output)
        output = self.fc2(output)
        output.retain_grad()
        s.append(output)

        if out_feature == False:
            return output, s, a
        else:
            return output,feature, s, a

class MNIST_AE(nn.Module):

    def __init__(self, dim_in):
        super(MNIST_AE, self).__init__()
        self.layer_input = nn.Linear(dim_in, 1000)
        self.layer_hidden1 = nn.Linear(1000, 500)
        self.layer_hidden2 = nn.Linear(500, 250)
        self.layer_hidden3 = nn.Linear(250, 30)
        self.layer_hidden4 = nn.Linear(30, 250)
        self.layer_hidden5 = nn.Linear(250, 500)
        self.layer_hidden6 = nn.Linear(500, 1000)
        self.layer_output = nn.Linear(1000, dim_in)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        # a and s are as defined in Algo1 of the Martens / Goose paper.  a and s would be a and h in the Goldfarb paper.
        # Notice that a and s indices are staggered.  List a would have indices starting from 0 and ends at L-1.
        # List s would start from layer index 1 and ends at L.
        a, s = [],[]

        input_shape = x.shape
        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        a.append(x)
        x = self.layer_input(x)
        x.retain_grad()
        s.append(x)
        x = self.activation(x)

        a.append(x)
        x = self.layer_hidden1(x)
        x.retain_grad()
        s.append(x)
        x = self.activation(x)

        a.append(x)
        x = self.layer_hidden2(x)
        x.retain_grad()
        s.append(x)
        x = self.activation(x)

        a.append(x)
        x = self.layer_hidden3(x)
        x.retain_grad()
        s.append(x)

        a.append(x)
        x = self.layer_hidden4(x)
        x.retain_grad()
        s.append(x)
        x = self.activation(x)

        a.append(x)
        x = self.layer_hidden5(x)
        x.retain_grad()
        s.append(x)
        x = self.activation(x)

        a.append(x)
        x = self.layer_hidden6(x)
        x.retain_grad()
        s.append(x)
        x = self.activation(x)

        a.append(x)
        x = self.layer_output(x)
        x.retain_grad()
        s.append(x)
        x = self.activation(x)

        return x.view(input_shape), s, a

class LSTM_reddit(nn.Module):

    def __init__(self, vocab_size=10004, embedding_size=96, hiddenLSTM_dim=670, hiddenLin1_dim=96, hiddenLin2_dim=10004):
        super(LSTM_reddit, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hiddenLSTM_dim)
        self.linear1 = nn.Linear(hiddenLSTM_dim, hiddenLin1_dim)
        self.linear1 = nn.Linear(hiddenLin1_dim, hiddenLin2_dim)
        self.tanh = nn.Tanh()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lin1_out = self.linear1(lstm_out.view(len(sentence), -1))
        # lin1_out = self.tanh(lin1_out)
        lin2_out = self.linear2(lin1_out)
        tag_scores = F.log_softmax(lin2_out, dim=1)
        return tag_scores


