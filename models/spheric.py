import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from functools import reduce

__all__ = [
	'Spherization',
	'SphericLinear', 'SphericConv2d',
	'SphericFNN', 'SphericCNN', 'SphericCNNVis3D'
]

eps = 1e-6
PI = 3.141592


class Spherization(nn.Module):

	def __init__(self, n_dims=None, 
				delta=1e-6,	scaling=1.0, radius=1.0, lrable=(True, False),
				range_type='bound', angle_type='quarter', sign_type='abs'):
		super(Spherization, self).__init__()

		if n_dims is None:
			raise Exception("'n_dims' is None. You have to initialize 'n_dims'.")

		radius = torch.tensor(radius, dtype=torch.float32)
		scaling = torch.tensor(scaling, dtype=torch.float32)
		delta = torch.tensor(delta, dtype=torch.float32)

        L = 0.01
        upper_bound = torch.tensor((PI / 2) * (1. - L), dtype=torch.float32)
        phi_L = torch.tensor(math.asin(delta ** (1. / n_dims)), dtype=torch.float32)
        phi_L = phi_L if phi_L < upper_bound else upper_bound

		W_theta = torch.diag(torch.ones(n_dims))
		W_theta = torch.cat((W_theta, W_theta[-1].unsqueeze(0)))

		W_phi = torch.ones((n_dims+1, n_dims+1))
		W_phi = torch.triu(W_phi, diagonal=1)
		W_phi[-2][-1] = 0.

		b_phi = torch.zeros(n_dims+1)
		b_phi[-1] = -PI / 2.

		self.register_buffer('phi_L', phi_L)
		self.register_buffer('W_theta', W_theta)
		self.register_buffer('W_phi', W_phi)
		self.register_buffer('b_phi', b_phi)
		if lrable[0]:
			self.scaling = nn.Parameter(scaling)
		else:
			self.register_buffer('scaling', scaling)
		if lrable[1]:
			self.radius = nn.Parameter(radius)
		else:
			self.register_buffer('radius', radius)
	
		self.register_buffer('delta', delta)
		
		self.angle_type = angle_type
		self.sign_type = sign_type

	def forward(self, x):
		x = self.spherize(x)
		return x

	def spherize(self, x):
		x_dim = x.dim()
		if x_dim == 4:
			n, c, h, w = x.shape
			x = x.permute(0, 2, 3, 1)
			x = x.reshape(-1, c)

		x = self.scaling * x
		x = self.angularize(x)
		x = torch.matmul(x, self.W_theta.T)

		v_sin = torch.sin(x)
		v_cos = torch.cos(x + self.b_phi)
		
		x = torch.matmul(torch.log(torch.abs(v_sin)+eps), self.W_phi) \
			+ torch.log(torch.abs(v_cos)+eps)
		x = self.radius * torch.exp(x)
	
		if x_dim == 4:
			x = x.reshape(n, h, w, c+1)
			x = x.permute(0, 3, 1, 2)

		return x

	def angularize(self, x):
		return (PI / 2 - self.phi_L) * torch.sigmoid(x) + self.phi_L


class SphericLinear(nn.Module):
	def __init__(self, in_features, out_features, bias=False,
				 delta=1e-6, scaling=1.0, radius=1.0, lrable=(True, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 pooling=None, batch_norm=False, input_shape=(1, ),
				 **kwargs):
		super(SphericLinear, self).__init__()

		self.sph = Spherization(n_dims=in_features, delta=delta, scaling=scaling, radius=radius, lrable=lrable,
								range_type=range_type, angle_type=angle_type, sign_type=sign_type)
		self.pooling = pooling
		shp = reduce(lambda a, b: a * b, input_shape)
		self.layer = nn.Linear((in_features+1)*shp, out_features, bias=False, **kwargs)

		self.batch_norm = batch_norm
		if self.batch_norm:
			self.bn = nn.BatchNorm1d(out_features)
	
	def forward(self, x):
		x = self.sph(x)

		if x.dim() == 4:
			if self.pooling is not None:
				x = self.pooling(x)

			n_features = reduce(lambda a, b: a * b, x.size()[1:])
			x = x.reshape(-1, n_features)

		x = self.layer(x)
		if self.batch_norm:
			x = self.bn(x)

		return x


class SphericConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=False,
				 delta=1e-6, scaling=1.0, radius=1.0, lrable=(True, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 pooling=None, batch_norm=False,
				 **kwargs):
		super(SphericConv2d, self).__init__()
		self.sph = Spherization(n_dims=in_channels, delta=delta, scaling=scaling, radius=radius, lrable=lrable,
								range_type=range_type, angle_type=angle_type, sign_type=sign_type)
		self.pooling = pooling
		self.layer = nn.Conv2d(in_channels+1, out_channels, kernel_size, bias=False, **kwargs)
	
		self.batch_norm = batch_norm
		if self.batch_norm:
			self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.sph(x)
		if self.pooling is not None:
			x = self.pooling(x)
		x = self.layer(x)
		if self.batch_norm:
			x = self.bn(x)

		return x


class SphericFNN(nn.Module):
	def __init__(self, n_dims=784,
				 delta=1e-6, scaling=1.0, radius=1.0, lrable=(True, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 **kwargs):
		super(SphericFNN, self).__init__()
		self.n_dims = n_dims

		self.fc1 = nn.Linear(self.n_dims, 500, bias=True)
		self.fc2 = nn.Linear(500, 300, bias=True)
		self.sph_fc3 = SphericLinear(300, 10, bias=False,
									 delta=delta, scaling=scaling, radius=radius, lrable=lrable,
									 range_type=range_type, angle_type=angle_type, sign_type=sign_type)

	def forward(self, x):
		if x.dim() > 2:
			x = x.view(-1, self.n_dims)
	
		x = self.fc1(x)
		x = F.relu(x)

		x = self.fc2(x)

		x = self.sph_fc3(x)

		return x


class SphericCNN(nn.Module):
	def __init__(self, n_dims=1,
				 delta=1e-6, scaling=1.0, radius=1.0, lrable=(True, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 **kwargs):
		super(SphericCNN, self).__init__()
		self.conv1 = nn.Conv2d(n_dims, 6, (5, 5), bias=True, padding=2)
		self.conv2 = nn.Conv2d(6, 16, (5, 5), bias=True, padding=0)
		self.fc3 = nn.Linear(16*5*5, 120, bias=True)
		self.fc4 = nn.Linear(120, 84, bias=True)
		self.sph_fc5 = SphericLinear(84, 10, bias=False,
									 delta=delta, scaling=scaling, radius=radius, lrable=lrable,
									 range_type=range_type, angle_type=angle_type, sign_type=sign_type)

	def forward(self, x):

		x = self.conv1(x)
		x = F.relu(x)
		x = F.max_pool2d(x, kernel_size=(2, 2))

		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, kernel_size=(2, 2))

		n_features = reduce(lambda a, b: a * b, x.size()[1:])
		x = x.reshape(-1, n_features)

		x = self.fc3(x)
		x = F.relu(x)

		x = self.fc4(x)

		x = self.sph_fc5(x)

		return x


class SphericCNNVis3D(nn.Module):
	def __init__(self, n_dims=1,
				 delta=1e-6, scaling=1.0, radius=1.0, lrable=(True, False),
				 range_type='bound', angle_type='quarter', sign_type='abs',
				 **kwargs):
		super(SphericCNNVis3D, self).__init__()
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
			SphericLinear(2, 10, bias=False,
						  delta=delta, scaling=scaling, radius=radius, lrable=lrable,
						  range_type=range_type, angle_type=angle_type, sign_type=sign_type)
		)

	def forward(self, x):
		x = self.conv_block(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
	
		return x


        
