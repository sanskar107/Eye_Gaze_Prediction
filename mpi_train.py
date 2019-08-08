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
from tensorboard_logger import configure, log_value


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

train_dataset = mpiloader('./data/', split="train")
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=1, pin_memory=True)


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
		if(i == 10):
			break
		cv2.imwrite('output/' + str(i) + 'real.png', imgs[i])
		cv2.imwrite('output/' + str(i) + 'recon.png', recon[i])



loss_fn = nn.MSELoss()
cos = nn.CosineSimilarity(dim = 1)

def update_lr(optimizer, epoch):
	if epoch == 10:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1
		print("LR Reduced")

	if epoch == 20:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1
		print("LR Reduced")

resume_training = True
freeze_fc = True
save_dir = 'weights/mpi/'
configure(save_dir)

if(resume_training):
	net.load_state_dict(torch.load('weights/unity/best_val.wts'))
	print("weights Loaded")

if(freeze_fc):
	for params in net.gaze.parameters():
		params.requires_grad = False

best_val_loss = 1

No_Epoch = 1000
print("Total Epochs : {} Iterations per Epoch : {}".format(No_Epoch, len(trainloader)))


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


for EPOCHS in range(No_Epoch):
	net.train()
	update_lr(optimizer, EPOCHS)
	running_loss = 0
	cosine_sim_train = 0
	for i, data in enumerate(trainloader):
		imgs, labels = data
		imgs, labels = imgs.to(device), labels.to(device)
		recon, gaze = net(imgs)

		loss_recon = loss_fn(recon, imgs)
		loss_gaze = loss_fn(gaze, labels)
		loss = loss_recon
		angle_error = torch.acos(cos(gaze, labels)).mean()*180.0/3.1415
		if(i % 10 == 0):
			print("epoch : {} iters : {} gaze : {:.3} recon : {:.3} angle_error : {:.3}".format(EPOCHS, i, loss_gaze.item(), loss_recon.item(), angle_error))
			log_value('angle_error', angle_error, int(EPOCHS*(len(trainloader)/10) + i/10))
			log_value('loss_gaze', loss_gaze.item(), int(EPOCHS*(len(trainloader)/10) + i/10))
			log_value('loss_recon', loss_recon.item(), int(EPOCHS*(len(trainloader)/10) + i/10))
		if(i == 0):
			save_img(imgs.cpu(), recon.cpu(), i)


		cosine_sim_train += torch.acos(cos(gaze, labels)).mean()*180.0/3.1415
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		running_loss += loss.item()
	train_loss = running_loss/len(trainloader)
	cosine_sim_train /= len(trainloader)

	print("EPOCH : {} Recon_Loss : {:.3f} Cosine_Sim_Train : {:.3f} ".format(EPOCHS, train_loss, cosine_sim_train))
	
	if train_loss < best_val_loss:
		torch.save(net.state_dict(), save_dir + 'best_val_mpi.wts')
		best_val_loss = train_loss
		print("Saved best loss weights")
	torch.save(net.state_dict(), save_dir + 'trained_mpi.wts')
