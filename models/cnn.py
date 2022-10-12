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

from .hsn import HSNLinear, HSNConv2d, SphericHSNLinear
from .spheric import SphericLinear, SphericConv2d


__all__ = ['HSNCNN9', 'HSNCNN9S', 'HSNCNN9D',
		   'SphericHSNCNN9S', 'SphericHSNCNN9D']


def make_layers(cfg: List[Union[str, int]],	batch_norm: bool = False) -> nn.Sequential:
	layers: List[nn.Module] = []
	in_channels = 3
	for idx, v in enumerate(cfg):
		if v == "M":
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			v = cast(int, v)
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	
	return nn.Sequential(*layers)


class CNN9(nn.Module):

	cnn9_cfg = [128, 128, 128, 'M', 192, 192, 192, 'M', 256, 256, 256, 'M']

	def __init__(self,
				 num_classes: int = 10,
				 **kwargs):
		super(CNN9, self).__init__()
		self.features = make_layers(self.cnn9_cfg, batch_norm=True)
		self.classifier = nn.Sequential(
			nn.Linear(256*4*4, 256),
			nn.Linear(256, num_classes),
		)
	
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x


def make_hsn_layers(cfg: List[Union[str, int]],
					batch_norm: bool = False,
					norm: str = 'none', w_norm: str = 'none') -> nn.Sequential:
	layers: List[nn.Module] = []
	in_channels = 3
	for idx, v in enumerate(cfg):
		if v == "M":
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			v = cast(int, v)
			conv2d = HSNConv2d(in_channels, v, operator=norm, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	
	return nn.Sequential(*layers)


class HSNCNN9(nn.Module):

	cnn9_cfg = [128, 128, 128, 'M', 192, 192, 192, 'M', 256, 256, 256, 'M']

	def __init__(self,
				 num_classes: int = 10,
				 norm: str = 'none',
				 w_norm: str = 'none',
				 **kwargs):
		super(HSNCNN9, self).__init__()
		self.features = make_hsn_layers(self.cnn9_cfg, batch_norm=True,
										norm=norm, w_norm=w_norm)
		self.classifier = nn.Sequential(
			HSNLinear(256*4*4, 256, operator=norm),
			HSNLinear(256, num_classes, operator=w_norm),
		)
	
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x


class HSNCNN9S(nn.Module):

	cnn9_cfg = [128, 128, 128, 'M', 192, 192, 192, 'M', 256, 256, 256, 'M']

	def __init__(self,
				 num_classes: int = 10,
				 norm: str = 'none',
				 w_norm: str = 'none',
				 **kwargs):
		super(HSNCNN9S, self).__init__()
		print(norm, w_norm)
		self.features = make_layers(self.cnn9_cfg, batch_norm=True)
		self.classifier = nn.Sequential(
			nn.Linear(256*4*4, 256),
			HSNLinear(256, num_classes, operator=norm)
		)
	
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x


class HSNCNN9D(nn.Module):

	cnn9_cfg = [128, 128, 128, 'M', 192, 192, 192, 'M', 256, 256, 256, 'M']

	def __init__(self,
				 num_classes: int = 10,
				 norm: str = 'none',
				 w_norm: str = 'none',
				 **kwargs):
		super(HSNCNN9D, self).__init__()
		print("HSNCNN9D", norm, w_norm)
		self.features = make_layers(self.cnn9_cfg, batch_norm=True)
		self.classifier = nn.Sequential(
			HSNLinear(256*4*4, 256, operator=norm),
			HSNLinear(256, num_classes, operator=w_norm)
		)
	
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x


class SphericHSNCNN9S(nn.Module):

	cnn9_cfg = [128, 128, 128, 'M', 192, 192, 192, 'M', 256, 256, 256, 'M']

	def __init__(self,
				 num_classes: int = 10,
				 delta=1e-6, radius=1.0, scaling=1.0, lrable=(False, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 norm: str = 'none', w_norm: str = 'none',
				 **kwargs):
		super(SphericHSNCNN9S, self).__init__()
		print("SphericHSNCNN9S")
		self.features = make_layers(self.cnn9_cfg, batch_norm=True)
		self.classifier = nn.Sequential(
			nn.Linear(256*4*4, 256),
			SphericHSNLinear(256, num_classes, operator=w_norm, spheric=False, bias=False,
						  delta=delta, radius=radius, scaling=scaling, lrable=lrable,
						  range_type=range_type, angle_type=angle_type, sign_type=sign_type)
		)
	
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x


class SphericHSNCNN9D(nn.Module):

	cnn9_cfg = [128, 128, 128, 'M', 192, 192, 192, 'M', 256, 256, 256, 'M']

	def __init__(self,
				 num_classes: int = 10,
				 delta=1e-6, radius=1.0, scaling=1.0, lrable=(False, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 norm: str = 'none', w_norm: str = 'none',
				 **kwargs):
		super(SphericHSNCNN9D, self).__init__()
		print("SphericHSNCNN9D", norm, w_norm, radius, scaling, lrable)
		self.features = make_layers(self.cnn9_cfg, batch_norm=True)
		self.classifier = nn.Sequential(
			SphericHSNLinear(256*4*4, 256, operator=norm, spheric=False, bias=False,
						  delta=delta, radius=radius, scaling=scaling, lrable=lrable,
						  range_type=range_type, angle_type=angle_type, sign_type=sign_type),
			HSNLinear(256, num_classes, operator=w_norm)
		)
	
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x

