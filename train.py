import sys
sys.path.append('/home/sanskar/Workspace/BTP/')
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from data.data_loader import *
from model_new import *
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

train_dataset = unityloader('./data/all/', split="train")
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True, num_workers=3, pin_memory=True)

val_dataset = unityloader('./data/all/', split="test")
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=500, shuffle=True, num_workers=3, pin_memory=True)


def save_img(imgs, recon, epoch):
	imgs = imgs.cpu().detach().numpy()
	recon = recon.cpu().detach().numpy()

	# imgs = imgs*img_stddev + img_mean
	# recon = recon*img_stddev + img_mean

	imgs = (imgs + 1.0)*255.0/2.0
	recon = (recon + 1.0)*255.0/2.0

	imgs = imgs.squeeze(1)
	recon = recon.squeeze(1)
	recon[recon < 0] = 0
	recon[recon > 255] = 255
	imgs = np.array(imgs, dtype = np.uint8)
	recon = np.array(recon, dtype = np.uint8)
	for i in range(imgs.shape[0]):
		if(i == 10):
			break
		cv2.imwrite('output/' + str(i) + 'real.png', imgs[i])
		cv2.imwrite('output/' + str(i) + 'recon.png', recon[i])


# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

loss_fn = nn.MSELoss()
cos = nn.CosineSimilarity(dim = 1)

def update_lr(optimizer, epoch):
	if (epoch + 1) % 3 == 0:
		lr = 0
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.85
			lr = param_group['lr']
		print("LR Reduced to {}".format(lr))


resume_training = False
freeze_fc = False
save_dir = 'weights/unity/'
if(resume_training):
	net.load_state_dict(torch.load(save_dir + 'best_val.wts'))
	print("weights Loaded")

if(freeze_fc):
	for params in net.gaze.parameters():
		params.requires_grad = False

best_val_loss = 10

No_Epoch = 1000
print("Total Epochs : {} Iterations per Epoch : {}".format(No_Epoch, len(trainloader)))


for EPOCHS in range(No_Epoch):
	net.train()
	update_lr(optimizer, EPOCHS)
	running_loss = 0
	cosine_sim_train = 0
	for i, data in enumerate(trainloader):
		imgs, labels = data
		imgs, labels = imgs.to(device), labels.to(device)
		recon, gaze, latent = net(imgs)

		if(i == 0):
			save_img(imgs, recon, EPOCHS)

		loss_recon = loss_fn(recon, imgs)
		loss_gaze = loss_fn(gaze, labels)
		angle_error = torch.acos(cos(gaze, labels)).mean()*180.0/3.1415
		loss = loss_recon + loss_gaze
		# loss = loss_recon + angle_error*0.1
		# if(i % 50 == 0):
		# 	save_img(imgs.cpu(), recon.cpu(), i)

		if(i % 10 == 0):
			print("Epoch : {} iter : {} gaze : {:.3} recon : {:.3} angle_error : {:.3}".format(EPOCHS, i, loss_gaze.item(), loss_recon.item(), angle_error.item()))

		cosine_sim_train += angle_error
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	
	net.eval()
	train_loss = running_loss/len(trainloader)
	cosine_sim_train /= len(trainloader)
	running_loss_recon = 0
	running_loss_gaze = 0
	cosine_sim_test = 0
	with torch.no_grad():
		for i, data in enumerate(valloader):
			imgs, labels = data
			imgs, labels = imgs.to(device), labels.to(device)

			recon, gaze, latent = net(imgs)
			loss_recon = loss_fn(recon, imgs)
			loss_gaze = loss_fn(gaze, labels)
			cosine_sim_test += torch.acos(cos(gaze, labels)).mean()*180.0/3.1415
			loss = loss_recon + loss_gaze

			if(i == 0):
				save_img(imgs.cpu(), recon.cpu(), i)
			running_loss_gaze += loss_gaze.item()
			running_loss_recon += loss_recon.item()
	running_loss_recon /= len(valloader)
	running_loss_gaze /= len(valloader)
	cosine_sim_test /= len(valloader)

	print("EPOCH : {} Loss_reconstruction : {:.3} Loss_Gaze : {:.3} Train_Loss : {:.3f} Cosine_Sim_Train : {:.3f} Cosine_Sim_Test : {:.3f}".format(EPOCHS, running_loss_recon, running_loss_gaze, train_loss, cosine_sim_train, cosine_sim_test))
	
	if cosine_sim_test < best_val_loss:
		torch.save(net.state_dict(), save_dir + 'best_val.wts')
		best_val_loss = cosine_sim_test
		print("Saved best loss weights")
	torch.save(net.state_dict(), save_dir + 'trained.wts')
