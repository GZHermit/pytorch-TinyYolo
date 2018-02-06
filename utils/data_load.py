# coding:utf-8
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from utils import datasets

def get_trainloader_sample():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='/home/gzh/Workspace/Dataset', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainloader


def get_trainloader():
    Data.TensorDataset