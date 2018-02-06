# coding:utf-8
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from models.tiny_yolo import TinyYolo

def train():
    # read the data

    # define the Network
    net = TinyYolo()
    net.train()
    net.cuda()

    # define the loss

    # define the optimizer

    # start to train the model

