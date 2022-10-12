"""
	ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import make_layers, cfgs, VGG
from typing import Union, List, Dict, Any, cast

import math
import numpy as np
from functools import reduce
from copy import deepcopy

if __name__=="__main__":
	from spheric import SphericLinear, SphericConv2d
else:
	from .spheric import SphericLinear, SphericConv2d


class SphericVGGS(VGG):
	def __init__(self, delta=1e-6, scaling=1.0, radius=1.0, lrable=(True, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 **kwargs):
		super(SphericVGGS, self).__init__(**kwargs)
		n_dims = self.features[-4].out_channels
		num_classes = self.classifier[-1].out_features
		self.classifier = nn.Sequential(
			nn.Linear(n_dims, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			SphericLinear(4096, num_classes, bias=False,
						  delta=delta, scaling=scaling, radius=radius, lrable=lrable,
						  range_type=range_type, angle_type=angle_type, sign_type=sign_type)
		)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x


def _sph_vggs(arch, cfg, batch_norm, widths=[64, 128, 256, 512], **kwargs):
	cfg_new = []
	idx = 0
	for v in cfgs[cfg]:
		if v == 'M':
			cfg_new.append('M')
			idx += 1
		else:
			idx = len(widths) - 1 if idx > len(widths) - 1 else idx
			cfg_new.append(widths[idx])
	
	return SphericVGGS(features=make_layers(cfg_new, batch_norm=batch_norm), **kwargs)


def sph_vgg11s_bn(widths=[64, 128, 256, 512], **kwargs):
	return _sph_vggs('vgg11s_bn', 'A', True, widths=widths, **kwargs)


def sph_vgg16s_bn(widths=[64, 128, 256, 512], **kwargs):
	return _sph_vggs('vgg16s_bn', 'D', True, widths=widths, **kwargs)


def sph_vgg19s_bn(widths=[64, 128, 256, 512], **kwargs):
	return _sph_vggs('vgg19s_bn', 'E', True, widths=widths, **kwargs)


class VGGW(VGG):
	def __init__(self, **kwargs):
		super(VGGW, self).__init__(**kwargs)
		n_dims = self.features[-4].out_channels
		num_classes = self.classifier[-1].out_features
		self.classifier = nn.Sequential(
			nn.Linear(n_dims, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)

		return x


def _vgg_w(arch, cfg, batch_norm, widths=[64, 128, 256, 512], **kwargs):
	cfg_new = []
	idx = 0
	for v in cfgs[cfg]:
		if v == 'M':
			cfg_new.append('M')
			idx += 1
		else:
			idx = len(widths) - 1 if idx > len(widths) - 1 else idx
			cfg_new.append(widths[idx])
	
	return VGGW(features=make_layers(cfg_new, batch_norm=batch_norm), **kwargs)


def vgg11_bn_w(widths=[64, 128, 256, 512], **kwargs):
	return _vgg_w('vgg11_bn_w', 'A', True, widths=widths, **kwargs)


def vgg16_bn_w(widths=[64, 128, 256, 512], **kwargs):
	return _vgg_w('vgg16_bn_w', 'D', True, widths=widths, **kwargs)


def vgg19_bn_w(widths=[64, 128, 256, 512], **kwargs):
	return _vgg_w('vgg19_bn_w', 'E', True, widths=widths, **kwargs)


def make_one_channel_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
	layers: List[nn.Module] = []
	in_channels = 1
	for v in cfg:
		if v == "M":
			layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
		else:
			v = cast(int, v)
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


def _vgg_one(arch, cfg, batch_norm, widths=[64, 128, 256, 512], **kwargs):
	cfg_new = []
	idx = 0
	for v in cfgs[cfg]:
		if v == 'M':
			cfg_new.append('M')
			idx += 1
		else:
			idx = len(widths) - 1 if idx > len(widths) - 1 else idx
			cfg_new.append(widths[idx])
	
	return VGGW(features=make_one_channel_layers(cfg_new, batch_norm=batch_norm), **kwargs)


def vgg11_bn_one(widths=[64, 128, 256, 512], **kwargs):
	return _vgg_one('vgg11_bn_one', 'A', True, widths=widths, **kwargs)


def _sph_vggs_one(arch, cfg, batch_norm, widths=[64, 128, 256, 512], **kwargs):
	cfg_new = []
	idx = 0
	for v in cfgs[cfg]:
		if v == 'M':
			cfg_new.append('M')
			idx += 1
		else:
			idx = len(widths) - 1 if idx > len(widths) - 1 else idx
			cfg_new.append(widths[idx])
	
	return SphericVGGS(features=make_one_channel_layers(cfg_new, batch_norm=batch_norm), **kwargs)


def sph_vgg11s_bn_one(widths=[64, 128, 256, 512], **kwargs):
	return _sph_vggs_one('sph_vgg11s_bn_one', 'A', True, widths=widths, **kwargs)

