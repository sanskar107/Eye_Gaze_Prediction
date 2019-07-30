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
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(102)

net = AutoEncoder()

net.to(device)
net.train()

img_mean = 108.9174
img_stddev = 48.6903

# img_mean = 0
# img_stddev = 255

params = net.state_dict()

train_dataset = eyeloader('./data/', split="train")
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1, pin_memory=True)

val_dataset = eyeloader('./data/', split="val")
valloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1, pin_memory=True)


def save_img(imgs, recon, epoch):
	imgs = imgs.detach().numpy()
	recon = recon.detach().numpy()
	imgs = imgs*img_stddev + img_mean
	recon = recon*img_stddev + img_mean
	imgs = imgs.squeeze(1)
	recon = recon.squeeze(1)
	imgs = np.array(imgs, dtype = np.uint8)
	recon = np.array(recon, dtype = np.uint8)
	recon[recon < 0] = 0
	recon[recon > 255] = 255
	for i in range(imgs.shape[0]):
		cv2.imwrite('output/' + str(i) + 'real.png', imgs[i])
		cv2.imwrite('output/' + str(i) + 'recon.png', recon[i])


# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

loss_fn = nn.MSELoss()

def update_lr(optimizer, epoch):
	if epoch == 25:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1

	if epoch == 40:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1

resume_training = True
if(resume_training):
	net.load_state_dict(torch.load('best_val.wts'))
	print("weights Loaded")

best_val_loss = 0.102

No_Epoch = 500
print("Total Epochs : {} Iterations per Epoch : {}".format(No_Epoch, len(trainloader)))

for EPOCHS in range(No_Epoch):
	net.train()
	# update_lr(optimizer, EPOCHS)
	running_loss = 0
	for i, data in enumerate(trainloader):
		imgs, labels = data
		imgs, labels = imgs.to(device), labels.to(device)
		recon, gaze = net(imgs)
		# print(recon)
		loss_recon = loss_fn(recon, imgs)
		loss_gaze = loss_fn(gaze, labels)
		loss = loss_recon + loss_gaze
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		running_loss += loss.item()
		# break
	net.eval()
	train_loss = running_loss/len(trainloader)
	running_loss_recon = 0
	running_loss_gaze = 0
	with torch.no_grad():
		for i, data in enumerate(valloader):
			imgs, labels = data
			imgs, labels = imgs.to(device), labels.to(device)
				
			recon, gaze = net(imgs)
			loss_recon = loss_fn(recon, imgs)
			loss_gaze = loss_fn(gaze, labels)
			loss = loss_recon + loss_gaze

			if(i == 0):
				save_img(imgs.cpu(), recon.cpu(), i)
			running_loss_gaze += loss_gaze.item()
			running_loss_recon += loss_recon.item()
	running_loss_recon /= len(valloader)
	running_loss_gaze /= len(valloader)

	print("EPOCH : {} Loss_reconstruction : {:.3} Loss_Gaze : {:.3} Train_Loss : {:.3f}".format(EPOCHS, running_loss_recon, running_loss_gaze, train_loss))
		
	if running_loss_recon < best_val_loss:
		torch.save(net.state_dict(), 'best_val.wts')
		best_val_loss = running_loss_recon
		print("Saved best loss weights")
	torch.save(net.state_dict(), 'trained.wts')
