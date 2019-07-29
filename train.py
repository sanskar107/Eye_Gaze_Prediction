import sys
sys.path.append('/home/sanskar/Workspace/BTP/')
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from data.data_loader import *
from model import *
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(102)

net = AutoEncoder()

net.to(device)
net.train()

params = net.state_dict()

train_dataset = eyeloader('./data/', split="train")
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1, pin_memory=True)

val_dataset = eyeloader('./data/', split="train")
valloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)

loss_fn = nn.MSELoss()

def update_lr(optimizer, epoch):
	if epoch == 25:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1

	if epoch == 40:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1

best_val_loss = 1e8

No_Epoch = 50
print("Total Epochs : {} Iterations per Epoch : {}".format(No_Epoch, len(trainloader)))

for EPOCHS in range(50):
	# net.train()
	update_lr(optimizer, EPOCHS)
	running_loss = 0
	for i, data in enumerate(trainloader):
		imgs, labels = data
		imgs, labels = imgs.to(device), labels.to(device)
		
		out = net(imgs)
		# loss = loss_fn(out, labels.float())
		loss = loss_fn(out, imgs)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		running_loss += loss.item()
		if i%100 == 99:
			print("EPOCH %d: %d/1000 loss=%f"%(EPOCHS, 2*(i+1), running_loss/len(trainloader)))
			running_loss = 0

	net.eval()
	running_loss = 0
	with torch.no_grad():
		for i, data in enumerate(valloader):
			imgs, labels = data
			imgs, labels = imgs.to(device), labels.to(device)
			
			out = net(imgs)
			loss = loss_fn(out, imgs)
			running_loss += loss.item()

	print("EPOCH %d: VAL loss=%f"%(EPOCHS, running_loss/len(valloader)))
	
	if running_loss < best_val_loss:
		torch.save(net.state_dict(), 'best_val.wts')
		best_val_loss = running_loss
	torch.save(net.state_dict(), 'trained.wts')
