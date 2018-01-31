import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class TinyYolo(nn.Module):
    def __init__(self):
        super(TinyYolo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=40, kernel_size=(1, 1), padding=1)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))), kernel_size=(2, 2), stride=2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))), kernel_size=(2, 2), stride=2)
        x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x))), kernel_size=(2, 2), stride=2)
        x = F.max_pool2d(F.leaky_relu(self.bn4(self.conv4(x))), kernel_size=(2, 2), stride=2)
        x = F.max_pool2d(F.leaky_relu(self.bn5(self.conv5(x))), kernel_size=(2, 2), stride=2)
        x = F.max_pool2d(F.leaky_relu(self.bn6(self.conv6(x))), kernel_size=(2, 2), stride=1)
        x = F.leaky_relu(self.bn7(self.conv7(x)))
        x = F.leaky_relu(self.bn8(self.conv8(x)))
        x = F.leaky_relu(self.conv9(x))
        return x
