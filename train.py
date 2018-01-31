# coding:utf-8
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from models.tinyyolo import TinyYolo


def train():
    # read the data
    data = Variable(torch.randn(64, 3, 416, 416))

    # define the Network
    net = TinyYolo()
    print(net.conv1)
    print(net)
    out = net(data)
    print(out.size())
    # optim.Adam
    net.zero_grad()
    out.backward()

    # params = list(net.parameters())
    # print('|===================|')
    # print(type(params[0]))
    # for param in net.parameters():
    #     print(type(param.data), param.size())
    #     # print(param)
    # print('|===================|')
    # net.zero_grad()
