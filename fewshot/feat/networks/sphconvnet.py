import torch.nn as nn

import sys
sys.path.insert(0, '..')
from models.spheric import Spherization
sys.path.insert(0, '.')
	

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, padding=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(),
		nn.MaxPool2d(2)
	)


class SphConvNet(nn.Module):

	def __init__(self, args, x_dim=3, hid_dim=64, z_dim=64):
		super().__init__()
		self.encoder = nn.Sequential(
			conv_block(x_dim, hid_dim),
			conv_block(hid_dim, hid_dim),
			conv_block(hid_dim, hid_dim),
			conv_block(hid_dim, z_dim),
		)
		self.conv = nn.Conv2d(z_dim, z_dim, kernel_size=5, stride=5)
		self.bn = nn.BatchNorm2d(z_dim)
		lrable = (True if args.lrable[0] == 'T' else False,
				  True if args.lrable[1] == 'T' else False)
		self.sph = Spherization(n_dims=z_dim,
								delta=args.delta, scaling=args.scaling, radius=args.radius, lrable=lrable)

	def forward(self, x):
		x = self.encoder(x)
#		x = nn.MaxPool2d(5)(x)
		x = self.conv(x)
		x = self.bn(x)
		x = x.view(x.size(0), -1)
		x = self.sph(x)
		return x

