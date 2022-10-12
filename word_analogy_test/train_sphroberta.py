import os
import copy
import time
import math
import datetime
import argparse
import numpy as np

import torch

from torch import nn, Tensor
from torch.utils.data import dataset, Dataset, DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from datasets import WikiDataset

import sys
sys.path.insert(0, '..')
import utils
from models.sphroberta import SphRobertaForMaskedLM

parser = argparse.ArgumentParser(description="Language Modeling")

parser.add_argument('--gpu-id', default=0, type=int)
parser.add_argument('--seed-num', default=1, type=int)
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--ckpt-dir', default='ckpt', type=str)
parser.add_argument('--ckpt-tag', default='sphroberta', type=str)
parser.add_argument('--pretrained', default='roberta-large', type=str)
parser.add_argument('--empty', action='store_true')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--log-interval', type=int, default=200)

sph = parser.add_argument_group('params - Spherization')
sph.add_argument('--delta', default=1e-6, type=float)
sph.add_argument('--radius', default=1.0, type=float)
sph.add_argument('--scaling', default=1.0, type=float)
sph.add_argument('--range-type', default='bound', choices=['bound', 'whole'], type=str)
sph.add_argument('--angle-type', default='quarter', choices=['quarter', 'half', 'full'], type=str)
sph.add_argument('--sign-type', default='abs', choices=['abs', 'cos', 'both'], type=str)
sph.add_argument('--init-weight', default='random', type=str,
				 choices=['random', 'uniform', 'xavier_uniform', 'kaiming_uniform', 
				 		  'normal', 'xavier_normal', 'kaiming_normal'])
sph.add_argument('--no-bias', action='store_true', default=False)
sph.add_argument('--ratio', default=0.0, type=float)
sph.add_argument('--lrable', default='FF', choices=['FF', 'FT', 'TF', 'TT'], type=str)

args = parser.parse_args()

utils.set_random_seed(seed_num=args.seed_num)
use_cuda = True if torch.cuda.is_available() and args.gpu_id != -1 else False
device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')

# Load and batch data
tokenizer = RobertaTokenizer.from_pretrained(args.pretrained)

tr_dataset = WikiDataset(tokenizer, ctg='train')
tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_dataset = WikiDataset(tokenizer, ctg='valid')
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
ts_dataset = WikiDataset(tokenizer, ctg='test')
ts_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Initiate an instance
config = RobertaConfig.from_pretrained(args.pretrained)
config = utils.update_config_sph_params(config, args)

if args.empty:
	net = SphRobertaForMaskedLM(config).to(device)
else:
	net = SphRobertaForMaskedLM.from_pretrained(args.pretrained, config=config).to(device)

# Run the model
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

def train(dataloader, epoch):
	net.train()
	tr_loss = 0.

	prev_time = time.time()
	for idx, data in enumerate(dataloader):

		optimizer.zero_grad()
	
		encoded_input = tokenizer(data, return_tensors='pt', 
								  padding='max_length', truncation=True)
		encoded_input = encoded_input.to(device)
		outputs = net(**encoded_input, labels=encoded_input['input_ids'])

		loss = outputs['loss']
		tr_loss += loss.item()
		loss.backward()

		optimizer.step()

		# verbose
		batches_done = (epoch - 1) * len(dataloader) + idx
		batches_left = args.n_epochs * len(dataloader) - batches_done
		time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
		prev_time = time.time()
		print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} (eta: {})".format(
			epoch, args.n_epochs, idx+1, len(dataloader), loss, time_left), end=' ')

	
	cnt = len(dataloader.dataset)
	tr_loss /= cnt
	
	return tr_loss


def test(dataloader):
	net.eval()
	ts_loss = 0.

	with torch.no_grad():
		for data in dataloader:

			encoded_input = tokenizer(data, return_tensors='pt', 
									  padding='max_length', truncation=True)
			encoded_input = encoded_input.to(device)
			outputs = net(**encoded_input, labels=encoded_input['input_ids'])

			loss = outputs['loss']
			ts_loss += loss.item()

	cnt = len(dataloader.dataset)
	ts_loss /= cnt

	return ts_loss


for epoch in range(1, args.n_epochs + 1):
	tr_loss = train(tr_dataloader, epoch)
	ts_loss = test(ts_dataloader)

	print("tr_loss: {:.4f}, ".format(tr_loss)
		+ "/ ts_loss: {:.4f}".format(ts_loss), end='')
print()

if args.save:
	os.makedirs(args.ckpt_dir, exist_ok=True)
	ckpt_filepath = os.path.join(args.ckpt_dir, args.ckpt_tag + '.pth')
	torch.save(net.state_dict(), ckpt_filepath)

