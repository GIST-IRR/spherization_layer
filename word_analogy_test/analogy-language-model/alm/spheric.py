import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from functools import reduce

__all__ = [
	'Spherization',
	'SphericLinear', 'SphericConv2d', 'SphericLSTMCell',
	'SphericFNN', 'SphericFNNS',
	'SphericCNN', 'SphericCNNS', 'SphericCNN3D'
]

BOUND = 1e-6
PI = math.pi   #PI = 3.141592


class Spherization(nn.Module):

	def __init__(self, n_dims=None, delta=1e-6,
				radius=1.0, scaling=1.0, range_type='bound'):
		super(Spherization, self).__init__()

		if n_dims is None:
			raise Exception("'n_dims' is None. You have to initialize 'n_dims'.")

		radius = torch.tensor(radius, dtype=torch.float32)
		scaling = torch.tensor(scaling, dtype=torch.float32)

		if range_type == 'whole':
			phi_L = torch.tensor(0., dtype=torch.float32)
		elif range_type == 'bound':
			L = 0.01
			upper_bound = (math.pi / 2) * (1. - L)
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
		self.register_buffer('radius', radius)
		self.register_buffer('scaling', scaling)

	def forward(self, x):
		x_dim = x.dim()
		if x_dim == 4:
			n, c, h, w = x.shape
			x = x.permute(0, 2, 3, 1)
			x = x.reshape(-1, c)

		x = self.scaling * x
		x = (PI / 2 - self.phi_L) * torch.sigmoid(x) + self.phi_L
		x = torch.matmul(x, self.W_theta.T)
		x = torch.matmul(torch.log(torch.sin(x)+BOUND), self.W_phi) \
			+ torch.log(torch.cos(x+self.b_phi)+BOUND)
		x = self.radius * torch.exp(x)
	
		if x_dim == 4:
			x = x.reshape(n, h, w, c+1)
			x = x.permute(0, 3, 1, 2)

		return x


class SphericLinear(nn.Module):
	def __init__(self, in_features, out_features, bias=False, delta=1e-6,
				 radius=1.0, scaling=1.0, range_type='bound',
				 max_pool=None, batch_norm=False, input_shape=(1, ),
				 **kwargs):
		super(SphericLinear, self).__init__()

		self.sph = Spherization(n_dims=in_features, delta=delta,
								radius=radius, scaling=scaling,	range_type=range_type)
		self.max_pool = max_pool
		shp = reduce(lambda a, b: a * b, input_shape)
		self.layer = nn.Linear((in_features+1)*shp, out_features, bias=False, **kwargs)

		self.batch_norm = batch_norm
		if self.batch_norm:
			self.bn = nn.BatchNorm1d(out_features)
	
	def forward(self, x):
		x = self.sph(x)

		if x.dim() == 4:
			if self.max_pool is not None:
				x = self.max_pool(x)

			n_features = reduce(lambda a, b: a * b, x.size()[1:])
			x = x.reshape(-1, n_features)

		x = self.layer(x)
		if self.batch_norm:
			x = self.bn(x)

		return x


class SphericConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=False, delta=1e-6,
				 radius=1.0, scaling=1.0, range_type='bound',
				 max_pool=None, batch_norm=False,
				 **kwargs):
		super(SphericConv2d, self).__init__()
		self.sph = Spherization(n_dims=in_channels, delta=delta,
								radius=radius, scaling=scaling,	range_type=range_type)
		self.max_pool = max_pool
		self.layer = nn.Conv2d(in_channels+1, out_channels, kernel_size, bias=False, **kwargs)
	
		self.batch_norm = batch_norm
		if self.batch_norm:
			self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.sph(x)
		if self.max_pool is not None:
			x = self.max_pool(x)
		x = self.layer(x)
		if self.batch_norm:
			x = self.bn(x)

		return x


class SphericLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, bias=False, delta=1e-6,
				 radius=1.0, scaling=1.0, range_type='bound',
				 **kwargs):
		super(SphericLSTMCell, self).__init__()
		self.sph = Spherization(n_dims=input_size, delta=delta,
								radius=radius, scaling=scaling, range_type=range_type)
		self.layer = nn.LSTMCell(input_size+1, hidden_size, bias=False)
	
	def forward(self, input, hx):
		sph_input = self.sph(input)
		input, hx = self.layer(sph_input, hx)
		
		return input, hx


class SphericFNN(nn.Module):
	def __init__(self, n_dims=784, delta=1e-6,
				 radius=1.0, scaling=1.0, range_type='bound',
				 **kwargs):
		super(SphericFNN, self).__init__()
		self.n_dims = n_dims

		self.fc1 = nn.Linear(self.n_dims, 500, bias=True)
		self.sph_fc2 = SphericLinear(500, 300, bias=False, delta=delta, 
									 radius=radius, scaling=scaling, range_type=range_type)
		self.sph_fc3 = SphericLinear(300, 10, bias=False, delta=delta, 
									 radius=radius, scaling=scaling, range_type=range_type)

	def forward(self, x):
		if x.dim() > 2:
			x = x.view(-1, self.n_dims)
	
		x = self.fc1(x)
		x = self.sph_fc2(x)
		x = self.sph_fc3(x)
		x = torch.sigmoid(x)

		return x


class SphericFNNS(nn.Module):
	def __init__(self, n_dims=784, delta=1e-6,
				 radius=1.0, scaling=1.0, range_type='bound',
				 **kwargs):
		super(SphericFNNS, self).__init__()
		self.n_dims = n_dims

		self.fc1 = nn.Linear(self.n_dims, 500, bias=True)
		self.fc2 = nn.Linear(500, 300, bias=True)
		self.sph_fc3 = SphericLinear(300, 10, bias=False, delta=delta,
									 radius=radius, scaling=scaling, range_type=range_type)

	def forward(self, x):
		if x.dim() > 2:
			x = x.view(-1, self.n_dims)
	
		x = self.fc1(x)
		x = torch.sigmoid(x)
		x = self.fc2(x)
		x = self.sph_fc3(x)
		x = torch.sigmoid(x)

		return x


class SphericCNN(nn.Module):
	def __init__(self, n_dims=1, delta=1e-6,
				 radius=1.0, scaling=1.0, range_type='bound',
				 **kwargs):
		super(SphericCNN, self).__init__()
		self.conv1 = nn.Conv2d(n_dims, 6, (5, 5), bias=True, padding=2)
		self.sph_conv2 = SphericConv2d(6, 16, (5, 5), bias=False, delta=delta,
									   radius=radius, scaling=scaling, range_type=range_type,
									   max_pool=nn.MaxPool2d((2, 2)),
									   padding=0)
		self.sph_fc3 = SphericLinear(16, 120, bias=False, delta=delta,
									 radius=radius, scaling=scaling, range_type=range_type,
									 max_pool=nn.MaxPool2d((2, 2)), input_shape=(5, 5))
		self.sph_fc4 = SphericLinear(120, 84, bias=False, delta=delta,
									 radius=radius, scaling=scaling, range_type=range_type)
		self.sph_fc5 = SphericLinear(84, 10, bias=False, delta=delta,
									 radius=radius, scaling=scaling, range_type=range_type)

	def forward(self, x):

		x = self.conv1(x)
		x = self.sph_conv2(x)
		x = self.sph_fc3(x)
		x = self.sph_fc4(x)
		x = self.sph_fc5(x)

		return x


class SphericCNNS(nn.Module):
	def __init__(self, n_dims=1, delta=1e-6,
				 radius=1.0, scaling=1.0, range_type='bound',
				 **kwargs):
		super(SphericCNNS, self).__init__()
		self.conv1 = nn.Conv2d(n_dims, 6, (5, 5), bias=True, padding=2)
		self.conv2 = nn.Conv2d(6, 16, (5, 5), bias=True, padding=0)
		self.fc3 = nn.Linear(16*5*5, 120, bias=True)
		self.fc4 = nn.Linear(120, 84, bias=True)
		self.sph_fc5 = SphericLinear(84, 10, bias=False, delta=delta,
									 radius=radius, scaling=scaling, range_type=range_type)

	def forward(self, x):

		x = torch.sigmoid(self.conv1(x))
		x = F.max_pool2d(x, kernel_size=(2, 2))

		x = torch.sigmoid(self.conv2(x))
		x = F.max_pool2d(x, kernel_size=(2, 2))

		n_features = reduce(lambda a, b: a * b, x.size()[1:])
		x = x.reshape(-1, n_features)

		x = torch.sigmoid(self.fc3(x))

		x = self.fc4(x)
		x = self.sph_fc5(x)

		return x


class SphericCNN3D(nn.Module):
	def __init__(self, n_dims=1,
				 delta=1e-6, radius=1.0, scaling=1.0,
				 range_type='bound', angle_type='quarter', sign_type='both',
				 **kwargs):
		super(SphericCNN3D, self).__init__()
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
						  delta=delta, radius=radius, scaling=scaling,
						  range_type=range_type, angle_type=angle_type, sign_type=sign_type)
		)

	def forward(self, x):
		x = self.conv_block(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		x = F.softmax(x, dim=1)
	
		return x
	

from transformers import BertPreTrainedModel, BertModel
class SphericBertForSequenceClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.bn = nn.BatchNorm1d(config.hidden_size)
		self.classifier = SphericLinear(config.hidden_size, self.config.num_labels)

#		self.init_weights()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
	):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		pooled_output = outputs[1]

		pooled_output = self.bn(pooled_output)
		logits = self.classifier(pooled_output)

		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			if self.num_labels == 1:
				#  We are doing regression
				loss_fct = nn.MSELoss()
				loss = loss_fct(logits.view(-1), labels.view(-1))
			else:
				loss_fct = nn.CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)

