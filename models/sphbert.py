import torch
from torch import nn
from transformers import (
	BertIntermediate,
	BertOutput,
	BertLayer,
	BertEncoder,
	BertModel,
	BertForMaskedLM,
	BertPreTrainedModel
)

if __name__=="__main__":
	from spheric import SphericLinear
else:
	from .spheric import SphericLinear

import sys
sys.path.insert(0, '..')
from utils import get_sph_params


class SphBertIntermediate(BertIntermediate):
	def __init__(self, config):
		super().__init__(config)
	
	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		return hidden_states


class SphBertOutput(BertOutput):
	def __init__(self, config):
		super().__init__(config)
		delta, radius, scaling, range_type, angle_type, sign_type = get_sph_params(config)
		self.dense = SphericLinear(config.intermediate_size, config.hidden_size, bias=False,
								   delta=delta, radius=radius, scaling=scaling, lrable=lrable,
								   range_type=range_type, angle_type=angle_type, sign_type=sign_type)
	

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
		

class SphericBertForSequenceClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		delta, radius, scaling, range_type, angle_type, sign_type = get_sph_params(config)

		n_units = int(config.hidden_size // 2) - 1
		self.classifier = nn.Sequential(
			nn.Linear(config.hidden_size, n_units),
			SphericLinear(n_units, config.num_labels, bias=False,
						  delta=delta, radius=radius, scaling=scaling, lrable=lrable,
						  range_type=range_type, angle_type=angle_type, sign_type=sign_type)
		)

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

		pooled_output = self.dropout(pooled_output)
		
		pre_actv = self.classifier[0](pooled_output)
		sph_actv = self.classifier[1].sph(pre_actv)
		logits = self.classifier[1].layer(sph_actv)

		outputs = outputs[2:] + (sph_actv.unsqueeze(1), )

		outputs = (logits,) + outputs  # add hidden states and attention if they are here

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


class SphericBertForSequenceClassificationV2(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = SphBertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)


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

		pooled_output = self.dropout(pooled_output)
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

