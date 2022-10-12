import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from functools import reduce

__all__ = ['SimpleFNN', 'LeNet5',
           'SimpleCNNVis3D', 'SimpleCNNVis2D',
           'SimpleCNNVis3DProj', 'SimpleCNNVis2DProj']


class SimpleFNN(nn.Module):
    def __init__(self, n_dims=784, bias=True, **kwargs):
        super(SimpleFNN, self).__init__()
        self.n_dims = n_dims

        self.fc1 = nn.Linear(self.n_dims, 500, bias=bias)
        self.fc2 = nn.Linear(500, 300, bias=bias)
        self.output = nn.Linear(300, 10, bias=bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(-1, self.n_dims)
    
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.output(x)

        return x


class LeNet5(nn.Module):
    def __init__(self, n_dims=1, bias=True, **kwarges):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(n_dims, 6, (5, 5), bias=bias, padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), bias=bias, padding=0)
        self.fc3 = nn.Linear(16*5*5, 120, bias=bias)
        self.fc4 = nn.Linear(120, 84, bias=bias)
        self.output = nn.Linear(84, 10, bias=bias)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = torch.flatten(x, 1)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.output(x)

        return x


class SimpleCNNVis3D(nn.Module):
    def __init__(self, n_dims=1, **kwargs):
        super(SimpleCNNVis3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_dims, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
    
        return x
        

class SimpleCNNVis2D(nn.Module):
    def __init__(self, n_dims=1, **kwargs):
        super(SimpleCNNVis2D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_dims, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.ReLU(inplace=True),
            nn.Linear(2, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
    
        return x
        

class SimpleCNNVis3DProj(nn.Module):
    def __init__(self, n_dims=1, **kwargs):
        super(SimpleCNNVis3DProj, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_dims, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.classifier[:-1](x)
        input_norm = torch.sqrt(torch.sum(x**2, dim=-1)+1e-4).reshape(-1, 1)
        x = x / input_norm
        x = self.classifier[-1](x)
#        x = x / input_norm
    
        return x
        
    def _get_input_norm(self, input):
        eps = 1e-4

        filt = nn.Linear(3, 10, bias=False)
        filt.weight = nn.Parameter(torch.ones(filt.weight.shape, dtype=torch.float32))
        filt = filt.to(input.device)
        filt.eval()

        input_norm = torch.sqrt(filt(input**2)+eps)
        return input_norm


class SimpleCNNVis2DProj(nn.Module):
    def __init__(self, n_dims=1, **kwargs):
        super(SimpleCNNVis2DProj, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_dims, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.ReLU(inplace=True),
            nn.Linear(2, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.classifier[:-1](x)
        input_norm = torch.sqrt(torch.sum(x**2, dim=-1)+1e-4).reshape(-1, 1)
        x = x / input_norm
        x = self.classifier[-1](x)
#        x = x / input_norm
    
        return x
        
    def _get_input_norm(self, input):
        eps = 1e-4

        filt = nn.Linear(2, 10, bias=False)
        filt.weight = nn.Parameter(torch.ones(filt.weight.shape, dtype=torch.float32))
        filt = filt.to(input.device)
        filt.eval()

        input_norm = torch.sqrt(filt(input**2)+eps)
        return input_norm

