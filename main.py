import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
import sys
from libs import data_io
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import pdb
import logging

parser = argparse.ArgumentParser(description='PyTorch Compound to Protein')
parser.add_argument('--data_dir', type=str,
                    help='path to train dataset')
parser.add_argument('--test', type=bool, default=False,
                    help='path to train dataset')
parser.add_argument('--resume', type=str, default=None,
                    help='path to latest checkpoint')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--test_data_dir', type=str, default='',
                    help='path to test dataset')
parser.add_argument('--wd', type=float, default=0.0005,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of total epochs to run')
parser.add_argument('--lr', type=float, default=0.001,
                    help='base learning rate')

logger = logging.getLogger()
fmt = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

file_handler = logging.FileHandler('train.log')
file_handler.setFormatter(fmt)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(fmt)
 
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)
class one_net(nn.Module):
	def __init__(self, c_in, p_in):
		super(one_net, self).__init__()
		self.fc1 = nn.Linear(c_in+p_in, 3000)
		self.bn1 = nn.BatchNorm1d(3000)
		self.fc2 = nn.Linear(3000, 3000)
		self.bn2 = nn.BatchNorm1d(3000)
		self.fc3 = nn.Linear(3000, 3000)
		self.bn3 = nn.BatchNorm1d(3000)
		self.fc4 = nn.Linear(3000, 3000)
		self.bn4 = nn.BatchNorm1d(3000)
		self.fc5 = nn.Linear(3000, 3000)
		self.bn5 = nn.BatchNorm1d(3000)
		self.fc_out = nn.Linear(3000,1)
	def forward(self, c, p):
		input_ = torch.cat((c, p), dim=1)
		h1 = F.dropout(self.bn1(F.relu(self.fc1(input_))))
		h2 = F.dropout(self.bn2(F.relu(self.fc2(h1))))
		h3 = F.dropout(self.bn3(F.relu(self.fc3(h2))))
		h4 = F.dropout(self.bn4(F.relu(self.fc4(h3))))
		h5 = F.dropout(self.bn5(F.relu(self.fc5(h4))))
		return F.sigmoid(self.fc_out(h5))


class layer(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.5, output=False):
        super(layer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.output = output
    def forward(self, x):
        if not self.output:
            return self.dropout(self.activation(self.bn(self.fc(x))))
        else:
            return self.fc(x)


class split_net(nn.Module):
	def __init__(self, c_in, p_in):
		super(split_net, self).__init__()
		self.cnet_layer1 = layer(c_in, 2000)
		self.cnet_layer2 = layer(2000 + 4000, 2000)
		self.cnet_layer3 = layer(2000 + 4000, 2000)
		self.pnet_layer1 = layer(p_in, 2000)
		self.pnet_layer2 = layer(2000 + 4000, 2000)
		self.pnet_layer3 = layer(2000 + 4000, 2000)
		self.mixnet_layer1 = layer(4000 + 4000, 1000)
		self.mixnet_output = layer(1000, 1, output=True)

		self.me = layer(4000, 4000)
	def forward(self, c, p):
		cnet_h1 = self.cnet_layer1(c)
		pnet_h1 = self.pnet_layer1(p)
		me_h1_in = torch.cat([cnet_h1, pnet_h1], dim=-1)
		me_h1_out = self.me(me_h1_in)
		
		cnet_h2 = self.cnet_layer2(torch.cat([cnet_h1, me_h1_out], dim=-1))
		pnet_h2 = self.pnet_layer2(torch.cat([pnet_h1, me_h1_out], dim=-1))
		me_h2_in = torch.cat([cnet_h2, pnet_h2], dim=-1)
		me_h2_out = self.me(me_h2_in)
		
		cnet_h3 = self.cnet_layer3(torch.cat([cnet_h2, me_h2_out], dim=-1))
		pnet_h3 = self.pnet_layer3(torch.cat([pnet_h2, me_h2_out], dim=-1))
		me_h3_in = torch.cat([cnet_h3, pnet_h3], dim=-1)
		me_h3_out = self.me(me_h3_in)


		mixnet_h1 = self.mixnet_layer1(torch.cat([cnet_h3, pnet_h3, me_h3_out], dim=-1))
		output = F.sigmoid(self.mixnet_output(mixnet_h1))
		return output


def save_fn(state, filename='c2p.pth.tar'):
    torch.save(state, filename)

def train(data_loader, eval_loader, model, epochs, optimizer):
    start = time.time()
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    best_acc = 0
    model.train()
    for epoch in range(args.start_epoch, epochs):
        lr_scheduler.step()
        for idx, (c, p, label) in enumerate(data_loader):
            c, p, label = c.cuda(), p.cuda(), label.cuda()
            optimizer.zero_grad()
            with torch.enable_grad():
                output = model(c, p)
                loss = F.binary_cross_entropy(output, label)
            loss.backward()
            optimizer.step()

            if idx%10 == 0:
                logger.info('Train Epoch:{}\tStep:{}/{}\tLoss:{:.6f}\tfscore:{:.6f}\tSec:{:.5f}'.format(
					epoch, idx, \
                    len(data_loader.dataset)//len(c),
					loss.data[0], \
                    accuracy_score(np.round(label.data.cpu().numpy()).astype(np.int), \
									(output.data.cpu().numpy()>0.5).astype(np.int), \
									), \
						time.time()-start))

                start = time.time()
        cur_acc = evalution_in_train(eval_loader, model)
        logger.info('Current acc:{:.6f}\tBest_acc:{:.6f}'.format(cur_acc, best_acc))
        if cur_acc > best_acc:
            save_fn({
				'epoch': epoch + 1,
	            'state_dict': model.state_dict(),
	            'optimizer' : optimizer.state_dict(),
				})
            best_acc = cur_acc

def evalution_in_train(data_loader, model):
	"""
	eval the model and return the best thresholds for pred
	"""
	data_pred, data_label = [], []

	model.eval()

	for idx, (c, p, label) in enumerate(data_loader):
		c = c.cuda()
		p = p.cuda()
		with torch.no_grad():
			output = model(c, p)
		pred = output.data.cpu().numpy()
		label = label.numpy()
		data_pred.append(pred)
		data_label.append(label)
	
	label = np.concatenate(data_label, axis=0)
	pred = np.concatenate(data_pred, axis=0)
	return 	accuracy_score(label.astype(np.int), \
						  (pred>0.5).astype(np.int))

def test(data_loader, model, thres):
	"""
	test the model and write to csv file
	"""
	data_pred = []
	model.eval()

	for idx, (c, p) in enumerate(data_loader):
		c = c.cuda()
		p = p.cuda()
		with torch.no_grad():
			output = model(c, p)
		pred = output.data.cpu().numpy()
		data_pred.append(pred)

	data_pred = np.concatenate(data_pred, axis=0)
	prob = pd.DataFrame(data_pred, index=False)
	prob.to_csv('./test_prob.csv')

def main():
	global args 
	args = parser.parse_args()
	# pdb.set_trace()
	train_dataset = data_io.data(args.data_dir, 'train')
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=64,
											   shuffle=True,
											   num_workers=8,
											   pin_memory=True,
											   drop_last=True)
	eval_dataset = data_io.data(args.data_dir, 'eval')
	eval_loader = torch.utils.data.DataLoader(eval_dataset,
											  batch_size=64,
											  shuffle=False,
											  num_workers=8,
											  pin_memory=True,
											  drop_last=True)

	if args.test:
		test_dataset = data_io.data(args.test_data_dir, 'test')
		test_loader = torch.utils.data.DataLoader(test_dataset,
												  batch_size=64,
												  shuffle=False,
												  num_workers=8,
												  pin_memory=True)
	net = split_net(2200, 1400)
	net.cuda()
	optimizer = optim.SGD(net.parameters(),
						  lr=args.lr,
						  weight_decay=args.wd,
						  momentum=0.9)

	if args.resume:
		if os.path.isfile(args.resume):
			logger.info("load checkpoint from '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			net.load_state_dict(checkpoint['state_dict'])
			if not args.test:
				optimizer.load_state_dict(checkpoint['optimizer'])
			logger.info("loaded checkpoint '{}' (epoch {})"
                 		 .format(args.resume, checkpoint['epoch']))
		else:
			logger.info("no checkpoint found at '{}'".format(args.resume))

	train(train_loader, eval_loader, net, epochs=args.epochs, optimizer=optimizer)
	if args.test:
		test(test_loader, net, best_thres)

if __name__ == '__main__':
	main()

