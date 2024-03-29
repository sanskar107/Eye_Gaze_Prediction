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
from tensorboard_logger import configure, log_value


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(102)

ae = AutoEncoder()
disc = Discriminator_wgan()
ae_unity = AutoEncoder()

ae.to(device)
ae_unity.to(device)
disc.to(device)

img_mean = 108.9174
img_stddev = 48.6903

# img_mean = 0
# img_stddev = 255

batch_size = 500

mpi_dataset = mpiloader('./data/', split="train")
mpiloader = torch.utils.data.DataLoader(mpi_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

unity_dataset = unityloader('./data/all/', split = "train")
unityloader = torch.utils.data.DataLoader(unity_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

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



cos = nn.CosineSimilarity(dim = 1)

def update_lr(optimizer, epoch):
	if (epoch + 1) % 3 == 0:
		lr = 0
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.85
			lr = param_group['lr']
		print("LR Reduced to {}".format(lr))


#Load unity weights
ae.load_state_dict(torch.load('weights/unity/best_val.wts'))
ae_unity.load_state_dict(torch.load('weights/unity/best_val.wts'))
print("Unity Weights Loaded")

#Freezing fc layers
for params in ae.gaze.parameters():
	params.requires_grad = False

resume_training = False
save_dir = 'weights/wgan/'

configure(save_dir)

if(resume_training):
	ae.load_state_dict(torch.load('weights/wgan/trained_ae.wts'))
	disc.load_state_dict(torch.load('weights/wgan/trained_disc.wts'))
	print("weights Loaded")


best_val_loss = 20

No_Epoch = 1000
print("Total Epochs : {} Iterations per Epoch : {}".format(No_Epoch, len(mpiloader)))


# optim_recn = torch.optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), lr=0.0001)
optim_gen = torch.optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), lr=0.0001)
optim_dis = torch.optim.Adam(filter(lambda p: p.requires_grad, disc.parameters()), lr=0.0001)

criterion_recn = nn.MSELoss() ####################CHange to RMSE
criterion_adverserial = nn.BCELoss()

for EPOCHS in range(No_Epoch):
	ae.train()
	disc.train()
	# update_lr(optimizer, EPOCHS)
	running_loss = 0
	cosine_sim_train = 0
	i = 0
	for data_unity, data_mpi in zip(unityloader, mpiloader):

		imgs_u, labels_u = data_unity
		imgs_m, labels_m = data_mpi
		imgs_u, labels_u = imgs_u.to(device), labels_u.to(device)
		imgs_m, labels_m = imgs_m.to(device), labels_m.to(device)

		recon_u, gaze_u, latent_u = ae_unity(imgs_u)
		recon_m, gaze_m, latent_m = ae(imgs_m)

		#Train Discriminator
		optim_dis.zero_grad()
		loss_disc = -torch.mean(disc(latent_u)) + torch.mean(disc(latent_m))
		loss_disc.backward(retain_graph = True)
		optim_dis.step()

		#Clip weights of discriminator
		for p in disc.parameters():
			p.data.clamp_(-0.01, 0.01)


		#Train generator every 5 iterations
		# if(i % 5 == 0):
		optim_gen.zero_grad()
		loss_adv = -torch.mean(disc(latent_m))
		loss_recn = criterion_recn(recon_m, imgs_m)
		loss_gen = loss_adv*0.1 + loss_recn*0.9
		loss_gen.backward()
		optim_gen.step()


		angle_error = torch.acos(cos(gaze_m, labels_m)).mean()*180.0/3.1415

		print("epoch : {} iters : {} loss_recn : {:.3} loss_adv : {:.3} loss_disc : {:.3} angle_error : {:.3}".format(EPOCHS, i, loss_recn.item(), loss_adv.item(), loss_disc.item(), angle_error))
		log_value('angle_error', angle_error, int(EPOCHS*(len(mpiloader)) + i))
		log_value('loss_recn', loss_recn.item(), int(EPOCHS*(len(mpiloader)) + i))
		log_value('loss_adv', loss_adv.item(), int(EPOCHS*(len(mpiloader)) + i))
		log_value('loss_disc', loss_disc.item(), int(EPOCHS*(len(mpiloader)) + i))
		if(i == 0):
			save_img(imgs_m.cpu(), recon_m.cpu(), i)

		i += 1
		cosine_sim_train += angle_error
	cosine_sim_train /= len(mpiloader)

	print("EPOCH : {} Cosine_Sim_Train : {:.3f} ".format(EPOCHS, cosine_sim_train))
	
	if cosine_sim_train < best_val_loss:
		torch.save(ae.state_dict(), save_dir + 'best_val_ae.wts')
		torch.save(disc.state_dict(), save_dir + 'best_val_disc.wts')
		best_val_loss = cosine_sim_train
		print("Saved best loss weights")
	torch.save(ae.state_dict(), save_dir + 'trained_ae.wts')
	torch.save(disc.state_dict(), save_dir + 'trained_disc.wts')
