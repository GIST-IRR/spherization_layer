import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import time
import datetime
import argparse
import numpy as np
from pathlib import Path

from utils import *
from models.spheric import SphericLinear


class Toy(nn.Module):
	def __init__(self):
		super(Toy, self).__init__()
		self.fc1 = nn.Linear(2, 2)
		self.fc2 = nn.Linear(2, 2)

	def forward(self, x):
	
		x = torch.sigmoid(self.fc1(x))
		x = self.fc2(x)

		return x

class ToySpheric(nn.Module):
	def __init__(self, device=None, delta=1e-6,
				 scaling=1.0, radius=1.0, lrable=(True, False)):
		super(ToySpheric, self).__init__()
		self.device = torch.device('cuda:0') if device is None else device
		self.fc1 = nn.Linear(2, 1, bias=False)
		self.sph_fc2 = SphericLinear(1, 2, delta=delta,
									 scaling=scaling, radius=radius, lrable=lrable)

	def forward(self, x):
	
		x = self.fc1(x)
		x = self.sph_fc2(x)

		return x


parser = argparse.ArgumentParser(description='Toy Task - A Simple Binary Image Classification')
parser.add_argument('--gpu-id', default=0, type=int)
parser.add_argument('--seed-num', default=1, type=int)
parser.add_argument('--net', default='spheric', choices=['spheric', 'baseline'], type=str)
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--res-dir', default='../result', type=str)
parser.add_argument('--res-tag', default='spheric', type=str)

hyper = parser.add_argument_group('params')
hyper.add_argument('--n_epochs', default=100, type=int)
hyper.add_argument('--lr', default=1e-2, type=float)
hyper.add_argument('--batch-size', default=16, type=int)

sph = parser.add_argument_group('params - Spherization')
sph.add_argument('--delta', default=1e-6, type=float)
sph.add_argument('--scaling', default=1.0, type=float)
sph.add_argument('--radius', default=1.0, type=float)
sph.add_argument('--lrable', default='TF', choices=['FF', 'FT', 'TF', 'TT'], type=str)

args = parser.parse_args()

set_random_seed(seed_num=args.seed_num)
device = torch.device('cuda:{}'.format(args.gpu_id))

x_0 = np.zeros((100, 2)) - (np.random.random((100, 2)) - 0.5) * 0.1
x_1 = np.ones((100, 2)) - (np.random.random((100, 2)) - 0.5) * 0.1

y_0 = torch.zeros(100)
y_1 = torch.ones(100)

x = torch.tensor(np.vstack((x_0, x_1)), dtype=torch.float32)
y = torch.cat((y_0, y_1)).long()

dataset = list(zip(x, y))
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

lrable = (True if args.lrable[0] == 'T' else False,
		  True if args.lrable[1] == 'T' else False)

if args.net == 'spheric':
	net = ToySpheric(delta=args.delta, scaling=args.scaling,
					 radius=args.radius, lrable=lrable)
	weights = torch.clone(net.sph_fc2.layer.weight.data.detach().cpu()).numpy()
else:
	net = Toy().to(device)
	w = torch.clone(net.fc2.weight.data.detach().cpu()).numpy()
	b = torch.clone(net.fc2.bias.data.detach().cpu()).numpy()
	weights = (w, b)

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr)

def train(dataloader, epoch):
	net.train()
	tr_loss = 0.
	correct = 0

	prev_time = time.time()
	for idx, (data, targets) in enumerate(dataloader):
		data, targets = data.to(device), targets.to(device)

		optimizer.zero_grad()

		output = net(data)

		loss = criterion(output, targets)
		tr_loss += loss.item()
		loss.backward()

		optimizer.step()

		pred = output.argmax(dim=1, keepdim=True)
		correct += pred.eq(targets.view_as(pred)).sum().item()

		# verbose
		batches_done = (epoch - 1) * len(dataloader) + idx
		batches_left = args.n_epochs * len(dataloader) - batches_done
		time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
		prev_time = time.time()
		print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} (eta: {})".format(
			epoch, args.n_epochs, idx+1, len(dataloader), loss, time_left), end=' ')
	
	cnt = len(dataloader.dataset)
	tr_loss /= cnt
	tr_acc = correct / cnt
	
	return tr_loss, tr_acc


def test(dataloader):
	net.eval()

	actv = []
	labels = []
	outputs = []

	with torch.no_grad():
		for data, targets in dataloader:
			data, targets = data.to(device), targets.to(device)

			x = net.fc1(data)
			if args.net == 'spheric':
				x1 = net.sph_fc2.sph(x)
				x2 = net.sph_fc2.layer(x1)
				output = F.softmax(x2, dim=1)
			else:
				x1 = torch.sigmoid(x)
				x2 = net.fc2(x1)
				output = F.softmax(x2, dim=1)

			actv.append(x1.detach().cpu().numpy())
			labels.append(targets.detach().cpu().numpy())
			outputs.append(output.detach().cpu().numpy())
	
	return actv, labels, outputs

if args.save:
	actv_init, _, _ = test(dataloader)
	actv_init = np.vstack(actv_init)

total_acc = []
for epoch in range(1, args.n_epochs + 1):
	tr_loss, tr_acc = train(dataloader, epoch)
	total_acc.append(tr_acc)
	print("loss: {:.4f}, acc: {:.4f} ".format(tr_loss, tr_acc), end='')
print()


if args.save:
	Path(args.res_dir).mkdir(parents=True, exist_ok=True)
	actv, labels, outputs = test(dataloader)

	actv = np.vstack(actv)
	labels = np.concatenate(labels)
	outputs = np.vstack(outputs)
	accs = np.array(total_acc)
	if args.net == 'spheric':
		hyperplane = net.sph_fc2.layer.weight.detach().cpu().numpy()
		theta_L = net.sph_fc2.sph.phi_L.detach().cpu().numpy()
	else:
		w = net.fc2.weight.detach().cpu().numpy()
		b = net.fc2.bias.detach().cpu().numpy()
		hyperplane = (w, b)
		theta_L = None

	filepath = Path(args.res_dir) / 'toy_{}.npy'.format(args.res_tag)
	np.save(filepath, (actv_init, actv, labels, outputs, accs, hyperplane, weights, theta_L))

