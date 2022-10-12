import torch
from torch import nn
from transformers import (
	RobertaIntermediate,
	RobertaOutput,
	RobertaLayer,
	RobertaEncoder,
	RobertaModel,
	RobertaForMaskedLM,
	RobertaPreTrainedModel
)

if __name__=="__main__":
	from spheric import SphericLinear
else:
	from .spheric import SphericLinear

import sys
sys.path.insert(0, '..')
from utils import get_sph_params


class SphRobertaIntermediate(RobertaIntermediate):
	def __init__(self, config):
		super().__init__(config)
	
	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		return hidden_states


class SphRobertaOutput(RobertaOutput):
	def __init__(self, config):
		super().__init__(config)
		delta, radius, scaling, range_type, angle_type, sign_type = get_sph_params(config)
		self.dense = SphericLinear(config.intermediate_size, config.hidden_size, bias=False,
								   delta=delta, radius=radius, scaling=scaling, lrable=lrable,
								   range_type=range_type, angle_type=angle_type, sign_type=sign_type)
	

class SphRobertaLayer(RobertaLayer):
	def __init__(self, config):
		super().__init__(config)
		self.intermediate = SphRobertaIntermediate(config)
		self.output = SphRobertaOutput(config)
	

class SphRobertaEncoder(RobertaEncoder):
	def __init__(self, config):
		super().__init__(config)
		self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers-1)]
								   + [SphRobertaLayer(config)])


class SphRobertaModel(RobertaModel):
	def __init__(self, config, add_pooling_layer=True):
		super().__init__(config)
		self.encoder = SphRobertaEncoder(config)


class SphRobertaForMaskedLM(RobertaForMaskedLM):
	def __init__(self, config):
		super().__init__(config)

		self.roberta = SphRobertaModel(config, add_pooling_layer=False)
	

if __name__=="__main__":
	from transformers import RobertaConfig
	config = RobertaConfig()
	sphroberta = SphRobertaForMaskedLM(config)
	print(sphroberta)
