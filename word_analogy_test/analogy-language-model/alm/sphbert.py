import torch
from torch import nn
from transformers import BertIntermediate, BertOutput, BertLayer, BertEncoder, BertModel, BertForMaskedLM

from .spheric import SphericLinear


class SphBertIntermediate(BertIntermediate):
	def __init__(self, config):
		super().__init__(config)
	
	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		return hidden_states


class SphBertOutput(BertOutput):
	def __init__(self, config):
		super().__init__(config)
		self.dense = SphericLinear(config.intermediate_size, config.hidden_size)
	

class SphBertLayer(BertLayer):
	def __init__(self, config):
		super().__init__(config)
		self.intermediate = SphBertIntermediate(config)
		self.output = SphBertOutput(config)
	

class SphBertEncoder(BertEncoder):
	def __init__(self, config):
		super().__init__(config)
		self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers-1)]
								   + [SphBertLayer(config)])


class SphBertModel(BertModel):
	def __init__(self, config, add_pooling_layer=True):
		super().__init__(config)
		self.encoder = SphBertEncoder(config)


class SphBertForMaskedLM(BertForMaskedLM):
	def __init__(self, config):
		super().__init__(config)

		self.bert = SphBertModel(config, add_pooling_layer=False)
		
