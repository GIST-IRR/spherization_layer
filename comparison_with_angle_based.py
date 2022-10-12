import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path

from models import *
import utils

NET = get_all_networks()

parser = argparse.ArgumentParser(description='Image Classification - CIFAR10')
parser.add_argument('--gpu-id', default=0, type=int)
parser.add_argument('--data-dir', default='/YOUR/DATA/DIRPATH', type=str)
parser.add_argument('--net', type=str, choices=list(NET.keys()))
parser.add_argument('--seed-num', default=1, type=int)

hyper = parser.add_argument_group('params')
hyper.add_argument('--lr', default=0.05, type=float)
hyper.add_argument('--n_epochs', default=100, type=int)
hyper.add_argument('--batch-size', default=128, type=int)
hyper.add_argument('--momentum', default=0.9, type=float)
hyper.add_argument('--weight-decay', default=5e-4, type=float)
hyper.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd'])
hyper.add_argument('--lr-scheduling', action='store_true')

sph = parser.add_argument_group('params - Spherization')
sph.add_argument('--delta', default=1e-6, type=float)
sph.add_argument('--radius', default=5.0, type=float)
sph.add_argument('--scaling', default=5.0, type=float)
sph.add_argument('--lrable', default='TF', choices=['FF', 'FT', 'TF', 'TT'], type=str)

hsn = parser.add_argument_group('params - HyperSphereNet')
hsn.add_argument('--norm', default=None, choices=['none', 'linear', 'cosine', 'sigmoid'], type=str)
hsn.add_argument('--w_norm', default=None, choices=['none', 'w_linear', 'w_cosine', 'w_sigmoid'], type=str)

args = parser.parse_args()

t_start = time.time()

utils.set_random_seed(seed_num=args.seed_num)
device = torch.device('cuda:{}'.format(args.gpu_id))

transform_tr = transforms.Compose([
					transforms.RandomHorizontalFlip(),
					transforms.RandomCrop(32, 4),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std =[0.229, 0.224, 0.225])
				])
transform_ts = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std =[0.229, 0.224, 0.225])
				])

trainset = CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_tr)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testset = CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_ts)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

lrable = (True if args.lrable[0] == 'T' else False,
		  True if args.lrable[1] == 'T' else False)

if args.net.startswith('sph'):
	net = NET[args.net](delta=args.delta, scaling=args.scaling, radius=args.radius, lrable=lrable)
elif args.net.startswith('hsnsph'):
	net = NET[args.net](norm=args.norm, w_norm=args.w_norm,
						delta=args.delta, scaling=args.scaling, radius=args.radius, lrable=lrable)
else:
	net = NET[args.net](norm=args.norm, w_norm=args.w_norm)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr,
					   weight_decay=args.weight_decay)


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
	ts_loss = 0.
	correct = 0

	with torch.no_grad():
		for data, targets in dataloader:
			data, targets = data.to(device), targets.to(device)

			output = net(data)

			loss = criterion(output, targets)
			ts_loss += loss.item()

			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(targets.view_as(pred)).sum().item()

	cnt = len(dataloader.dataset)
	ts_loss /= cnt
	ts_acc = correct / cnt

	return ts_loss, ts_acc


lr = args.lr
for epoch in range(1, args.n_epochs + 1):
	if epoch in [80, 130]:
		lr = utils.lr_decay(optimizer, lr, decay_rate=0.1)

	tr_loss, tr_acc = train(trainloader, epoch)
	ts_loss, ts_acc = test(testloader)

	if args.net.startswith('sph') or args.net.startswith('hsnsph'):
		layer_idx = -2 if args.net[-1] == 'd' else -1
		s = net.classifier[layer_idx].sph.scaling
		r = net.classifier[layer_idx].sph.radius
		print("loss: {:.4f}, acc: {:.4f} ".format(tr_loss, tr_acc)
			+ "/ test_loss: {:.4f}, test_acc: {:.4f} ".format(ts_loss, ts_acc)
			+ "/ s: {:.4f} r: {:.4f}".format(s, r), end='')
	else:
		print("loss: {:.4f}, acc: {:.4f} ".format(tr_loss, tr_acc)
			+ "/ test_loss: {:.4f}, test_acc: {:.4f}".format(ts_loss, ts_acc), end='')

elapsed_time = time.time() - t_start
print("\n[ Elapsed Time: {:.4f} ]".format(elapsed_time))

