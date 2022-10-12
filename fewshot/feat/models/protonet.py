import torch
import torch.nn as nn
from feat.utils import euclidean_metric, cosine_similarity


class ProtoNet(nn.Module):

	def __init__(self, args):
		super().__init__()
		self.args = args
		if args.model_type == 'ConvNet':
			from feat.networks.convnet import ConvNet
			self.encoder = ConvNet()
		elif args.model_type == 'ResNet':
			from feat.networks.resnet import ResNet
			self.encoder = ResNet()
		elif args.model_type == 'AmdimNet':
			from feat.networks.amdimnet import AmdimNet
			self.encoder = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
		elif args.model_type == 'SphConvNet':
			from feat.networks.sphconvnet import SphConvNet
			self.encoder = SphConvNet(args)
		elif args.model_type == 'SphResNet':
			from feat.networks.sphresnet import SphResNet
			self.encoder = SphResNet(args)
		else:
			raise ValueError('')

	def forward(self, data_shot, data_query):
		proto = self.encoder(data_shot)
		proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
		if self.args.metric == 'cos':
			logits = 1000. * cosine_similarity(self.encoder(data_query), proto) / self.args.temperature
		else:
			logits = euclidean_metric(self.encoder(data_query), proto) / self.args.temperature

		return logits
