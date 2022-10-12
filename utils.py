import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
import random
import numpy as np

BOUND = 1e-6
PI = math.pi


def set_random_seed(seed_num=1):
	random.seed(seed_num)
	np.random.seed(seed_num)
	torch.manual_seed(seed_num)
	torch.cuda.manual_seed(seed_num)
	torch.cuda.manual_seed_all(seed_num)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def lr_scheduling(optimizer, epoch, lr):
	"""
		(ref) https://github.com/chengyangfu/pytorch-vgg-cifar10
	"""
	lr = lr * (0.5 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	
	return lr


def lr_decay(optimizer, lr, decay_rate=.2):
	"""
		(ref) https://github.com/weiaicunzai/pytorch-cifar100
	"""
	lr = lr * decay_rate
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	
	return lr


def update_config_sph_params(config, args):
	setattr(config, 'delta', args.delta)
	setattr(config, 'radius', args.radius)
	setattr(config, 'scaling', args.scaling)
	setattr(config, 'range_type', args.range_type)
	setattr(config, 'angle_type', args.angle_type)
	setattr(config, 'sign_type', args.sign_type)

	return config


def get_sph_params(config):
	delta = config.delta if hasattr(config, 'delta') else 1.0
	radius = config.radius if hasattr(config, 'radius') else 1.0
	scaling = config.scaling if hasattr(config, 'scaling') else 1.0
	range_type = config.range_type if hasattr(config, 'range_type') else 'bound'
	angle_type = config.angle_type if hasattr(config, 'angle_type') else 'quarter'
	sign_type = config.sign_type if hasattr(config, 'sign_type') else 'abs'

	return delta, radius, scaling, range_type, angle_type, sign_type

