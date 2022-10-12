from .basic import *
from .spheric import *
from .vgg import *
from .cnn import *


def get_all_networks():
	return {
		'simple_fnn': SimpleFNN,
		'sph_fnn': SphericFNN,
		'lenet': LeNet5,
		'simple_cnn_2d': SimpleCNNVis2D,
		'simple_cnn_3d': SimpleCNNVis3D,
		'proj_cnn_2d': SimpleCNNVis2DProj,
		'proj_cnn_3d': SimpleCNNVis3DProj,
		'sph_cnn': SphericCNN,
		'sph_cnn_3d': SphericCNNVis3D,
		'vgg11_bn_w': vgg11_bn_w,
		'vgg11_bn_one': vgg11_bn_one,
		'vgg16_bn_w': vgg16_bn_w,
		'vgg19_bn_w': vgg19_bn_w,
		'sph_vgg11s_bn': sph_vgg11s_bn,
		'sph_vgg11s_bn_one': sph_vgg11s_bn_one,
		'sph_vgg16s_bn': sph_vgg16s_bn,
		'sph_vgg19s_bn': sph_vgg19s_bn,
		'hsn_cnn9': HSNCNN9,
		'hsn_cnn9-s': HSNCNN9S,
		'hsn_cnn9-d': HSNCNN9D,
		'hsnsph_cnn9-s': SphericHSNCNN9S,
		'hsnsph_cnn9-d': SphericHSNCNN9D,
	}

