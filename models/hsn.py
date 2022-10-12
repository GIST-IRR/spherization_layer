"""
    (ref) https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Tuple, Union, Dict, Any, cast
from torchvision.models.vgg import make_layers, cfgs, VGG

import math
import numpy as np

from .spheric import Spherization


__all__ = ['HSNLinear', 'HSNConv2d',
           'SphericHSNLinear']

PI = 3.141592


class HSNOpr(nn.Module):
    def __init__(self, operator: str = 'none', spheric: bool = False):
        super(HSNOpr, self).__init__()
        self.operator = operator
        self.spheric = spheric

    def _get_filter_norm(self):
        pass

    def _get_input_norm(self, input):
        pass

    def add_operator(self, x, input):
        eps = 1e-4
        upper_bound = 1. - eps
        lower_bound = -1. + eps
        if self.operator != 'none':
            if not self.spheric:
                wnorm = self._get_filter_norm()
            if not self.operator.startswith('w'):
                xnorm = self._get_input_norm(input)

        if self.operator == 'w_linear':
            if not self.spheric:
                x = x / wnorm
            x[x >= upper_bound] = x[x >= upper_bound] / (x[x >= upper_bound] + eps) - eps
            x[x <= lower_bound] = x[x <= lower_bound] / (x[x <= lower_bound] - eps) * -1. + eps
            x = -0.63662 * torch.acos(x) + 1.
        elif self.operator == 'w_cosine':
            if not self.spheric:
                x = x / wnorm
        elif self.operator == 'w_sigmoid':
            k_value_w = 0.3
            constant_coeff_w = (1. + math.exp(-PI/(2*k_value_w)))/(1. - math.exp(-PI/(2*k_value_w)))
            if not self.spheric:
                x = x / wnorm
            x[x >= upper_bound] = x[x >= upper_bound] / (x[x >= upper_bound] + eps) - eps
            x[x <= lower_bound] = x[x <= lower_bound] / (x[x <= lower_bound] - eps) * -1. + eps
            x = constant_coeff_w * \
                (1. - torch.exp(torch.acos(x)/k_value_w-PI/(2*k_value_w))) / \
                (1. + torch.exp(torch.acos(x)/k_value_w-PI/(2*k_value_w)))

        if self.operator == 'linear':
            x = x / xnorm
            x = x / wnorm
            x[x >= upper_bound] = upper_bound
            x[x <= lower_bound] = lower_bound
            x = -0.63662 * torch.acos(x) + 1.
        elif self.operator == 'cosine':
            x = x / xnorm
            x = x / wnorm
        elif self.operator == 'sigmoid':
            k_value = 0.3
            constant_coeff = (1. + math.exp(-PI/(2*k_value)))/(1. - math.exp(-PI/(2*k_value)))
            x = x / xnorm
            x = x / wnorm
            x[x >= upper_bound] = upper_bound
            x[x <= lower_bound] = lower_bound
            x = constant_coeff * \
                (1. - torch.exp(torch.acos(x)/k_value-PI/(2*k_value))) / \
                (1. + torch.exp(torch.acos(x)/k_value-PI/(2*k_value)))
        
        return x


class HSNLinear(HSNOpr):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 operator: str = 'none',
                 spheric: bool = False,
                 **kwargs):
        super(HSNLinear, self).__init__(operator)
        bias = False if operator.startswith('w') else True
        self.fc = nn.Linear(in_features, out_features, bias=bias, **kwargs)
    
    def _get_filter_norm(self):
        eps = 1e-4
        filter_norm = torch.sqrt(torch.sum(self.fc.weight ** 2, dim=1, keepdim=True)+eps).T
        return filter_norm
    
    def _get_input_norm(self, input):
        eps = 1e-4

        filt = nn.Linear(self.fc.in_features, self.fc.out_features, bias=False)
        filt.weight = nn.Parameter(torch.ones(filt.weight.shape, dtype=torch.float32))
        filt = filt.to(input.device)
        filt.eval()

        input_norm = torch.sqrt(filt(input**2)+eps)
        return input_norm

    def forward(self, x):
        eps = 1e-4
        upper_bound = 1. - eps
        lower_bound = -1. + eps

        wnorm = self._get_filter_norm()
        xnorm = torch.sqrt(torch.sum(x**2, dim=-1)+eps).reshape(-1, 1)

        x = x / xnorm
        x = self.fc(x)

        if self.operator == 'w_linear':
            x = x * xnorm
            x = x / wnorm
            x[x >= upper_bound] = x[x >= upper_bound] / (x[x >= upper_bound] + eps) - eps
            x[x <= lower_bound] = x[x <= lower_bound] / (x[x <= lower_bound] - eps) * -1. + eps
            x = -0.63662 * torch.acos(x) + 1.
        elif self.operator == 'w_cosine':
            x = x * xnorm
            x = x / wnorm
        elif self.operator == 'w_sigmoid':
            k_value_w = 0.3
            constant_coeff_w = (1. + math.exp(-PI/(2*k_value_w)))/(1. - math.exp(-PI/(2*k_value_w)))
            x = x * xnorm
            x = x / wnorm
            x[x >= upper_bound] = x[x >= upper_bound] / (x[x >= upper_bound] + eps) - eps
            x[x <= lower_bound] = x[x <= lower_bound] / (x[x <= lower_bound] - eps) * -1. + eps
            x = constant_coeff_w * \
                (1. - torch.exp(torch.acos(x)/k_value_w-PI/(2*k_value_w))) / \
                (1. + torch.exp(torch.acos(x)/k_value_w-PI/(2*k_value_w)))

        if self.operator == 'linear':
            x = x / wnorm
            x[x >= upper_bound] = upper_bound
            x[x <= lower_bound] = lower_bound
            x = -0.63662 * torch.acos(x) + 1.
        elif self.operator == 'cosine':
            x = x / wnorm
        elif self.operator == 'sigmoid':
            k_value = 0.3
            constant_coeff = (1. + math.exp(-PI/(2*k_value)))/(1. - math.exp(-PI/(2*k_value)))
            x = x / wnorm
            x[x >= upper_bound] = upper_bound
            x[x <= lower_bound] = lower_bound
            x = constant_coeff * \
                (1. - torch.exp(torch.acos(x)/k_value-PI/(2*k_value))) / \
                (1. + torch.exp(torch.acos(x)/k_value-PI/(2*k_value)))
        
        return x


class HSNConv2d(HSNOpr):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 operator: str = 'none',
                 spheric: bool = False,
                 **kwargs):
        super(HSNConv2d, self).__init__(operator)
        bias = False if operator.startswith('w') else True
        self.conv = nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        
    def _get_filter_norm(self):
        eps = 1e-4
        filter_norm = torch.sqrt(torch.sum(self.conv.weight ** 2, dim=(1, 2, 3), keepdim=True)+eps).permute(1, 0, 2, 3)
        return filter_norm
    
    def _get_input_norm(self, input):
        eps = 1e-4

        filt = nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                         kernel_size=self.conv.kernel_size,
                         stride=self.conv.stride,
                         padding=self.conv.padding,
                         bias=False)
        filt.weight = nn.Parameter(torch.ones(filt.weight.shape, dtype=torch.float32))
        filt = filt.to(input.device)
        filt.eval()

        input_norm = torch.sqrt(filt(input**2)+eps)
        return input_norm
    
    def forward(self, input):
        x = self.conv(input)
        x = self.add_operator(x, input)
        return x


class SphericHSNLinear(HSNOpr):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 operator: str = 'none',
                 spheric: bool = False,
                 delta=1e-6, radius=1.0, scaling=1.0, lrable=(False, False),
                 range_type='bound', angle_type='quarter', sign_type='abs',
                 **kwargs):
        super(SphericHSNLinear, self).__init__(operator)
        self.sph = Spherization(n_dims=in_features, delta=delta, radius=radius, scaling=scaling, lrable=lrable,
                                range_type=range_type, angle_type=angle_type, sign_type=sign_type)
        self.fc = nn.Linear(in_features+1, out_features, **kwargs)
    
    def _get_filter_norm(self):
        eps = 1e-4
        filter_norm = torch.sqrt(torch.sum(self.fc.weight ** 2, dim=1, keepdim=True)+eps).T
        return filter_norm
    
    def _get_input_norm(self, input):
        eps = 1e-4

        filt = nn.Linear(self.fc.in_features, self.fc.out_features, bias=False)
        filt.weight = nn.Parameter(torch.ones(filt.weight.shape, dtype=torch.float32))
        filt = filt.to(input.device)
        filt.eval()

        input_norm = torch.sqrt(filt(input**2)+eps)
        return input_norm
    
    def forward(self, input):
        x_sph = self.sph(input)
        x = self.fc(x_sph)
        x = self.add_operator(x, x_sph)
        return x
