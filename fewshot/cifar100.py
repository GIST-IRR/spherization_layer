import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time, datetime
import argparse
import numpy as np
from pathlib import Path

import utils

NET = {
	'resnet18': resnet18,
	'resnet34': resnet34,
	'resnet50': resnet50
}

parser = argparse.ArgumentParser(description='Image Classification - CIFAR100')
parser.add_argument('--gpu-id', default=0, type=int)
parser.add_argument('--data-dir', default='/YOUR/DATA/DIRPATH', type=str)
parser.add_argument('--net', type=str, choices=list(NET.keys()))
parser.add_argument('--seed-num', default=1, type=int)
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--res-dir', default='result', type=str)
parser.add_argument('--res-tag', default='resnet18', type=str)
parser.add_argument('--ckpt-dir', default='ckpt', type=str)

hyper = parser.add_argument_group('params')
hyper.add_argument('--lr', default=0.1, type=float)
hyper.add_argument('--step', default=20, type=int)
hyper.add_argument('--decay-rate', default=.1, type=float)
hyper.add_argument('--n_epochs', default=200, type=int)
hyper.add_argument('--batch-size', default=128, type=int)
hyper.add_argument('--momentum', default=0.9, type=float)
hyper.add_argument('--weight-decay', default=5e-4, type=float)

args = parser.parse_args()

t_start = time.time()

utils.set_random_seed(seed_num=args.seed_num)
device = torch.device('cuda:{}'.format(args.gpu_id))

transform_tr = transforms.Compose([
#					transforms.RandomCrop(32, padding=4),
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.RandomRotation(15),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
										 std =[0.2673, 0.2564, 0.2762])
				])
transform_ts = transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
										 std =[0.2673, 0.2564, 0.2762])
				])

trainset = CIFAR100(root=args.data_dir, train=True, download=False, transform=transform_tr)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testset = CIFAR100(root=args.data_dir, train=False, download=False, transform=transform_ts)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

net = NET[args.net]()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
					  momentum=args.momentum,
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


if args.save:
	result = [] # [(tr_loss, tr_acc, ts_loss, ts_acc)]

lr = args.lr
for epoch in range(1, args.n_epochs + 1):
	if epoch in [60, 120, 160]:
		lr = utils.lr_decay(optimizer, lr)

	tr_loss, tr_acc = train(trainloader, epoch)
	ts_loss, ts_acc = test(testloader)

	print("loss: {:.4f}, acc: {:.4f} ".format(tr_loss, tr_acc)
		+ "/ test_loss: {:.4f}, test_acc: {:.4f}".format(ts_loss, ts_acc), end='')

	if args.save:
		result.append([tr_loss, tr_acc, ts_loss, ts_acc])

print("\n[ Elapsed Time: {:.4f} ]".format(time.time() - t_start))

if args.save:
	Path(args.res_dir).mkdir(parents=True, exist_ok=True)
	res_filepath = Path(args.res_dir) / 'result_{}.npy'.format(args.res_tag)
	np.save(res_filepath, np.array(result))
	Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
	ckpt_filepath = Path(args.ckpt_dir) / 'ckpt_{}.pth'.format(args.res_tag)
	torch.save(net.state_dict(), ckpt_filepath)


