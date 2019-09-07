import sys
sys.path.append('~/eyegaze/')
import torch
import torch.nn as nn
from torch import autograd
import numpy as np
from torchvision import datasets, models, transforms
from data.data_loader import *
from model import *
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import math
import cv2
# from tensorboard_logger import configure, log_value


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

batch_size = 512
gamma = 10

mpi_dataset_test = mpiloader('./data/', split="test")
mpiloader_test = torch.utils.data.DataLoader(mpi_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

mpi_dataset = mpiloader('./data/', split="train")
mpiloader = torch.utils.data.DataLoader(mpi_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

unity_dataset = unityloader('./data/all/', split = "train")
unityloader = torch.utils.data.DataLoader(unity_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)



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


#Load unity weights
ae_unity.load_state_dict(torch.load('weights/unity_new/best_val.wts'))
ae.load_state_dict(torch.load('weights/mpi/best_val.wts'))
disc.load_state_dict(torch.load('weights/gan/disc_only.wts'))
print("Unity and MPI Weights Loaded")

#Freezing fc layers
for params in ae.gaze.parameters():
	params.requires_grad = False

resume_training = True
save_dir = 'weights/wgan_gp/'

# configure(save_dir)

if(resume_training):
	ae.load_state_dict(torch.load('weights/wgan_gp/best_val_ae.wts'))
	disc.load_state_dict(torch.load('weights/wgan_gp/best_val_disc.wts'))
	print("weights Loaded")


best_val_loss = 8.85

No_Epoch = 1000

len_loader = min(len(unityloader), len(mpiloader))

print("Total Epochs : {} Iterations per Epoch : {}".format(No_Epoch, len_loader))


# optim_recn = torch.optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), lr=0.0001)
optim_gen = torch.optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), lr=0.0001)
optim_dis = torch.optim.Adam(filter(lambda p: p.requires_grad, disc.parameters()), lr=0.0001)

criterion_recn = nn.MSELoss(reduction = "sum") ####################CHange to RMSE
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

		if(imgs_m.shape[0] != imgs_u.shape[0]):
			continue

		recon_u, gaze_u, latent_u = ae_unity(imgs_u)
		recon_m, gaze_m, latent_m = ae(imgs_m)

		_, test_u, _ = ae_unity(imgs_u)
		print(torch.acos(cos(test_u, labels_u)).mean()*180.0/3.1415)

		#Train Discriminator
		optim_dis.zero_grad()

		# epsilon = torch.rand(batch_size,1)
		# epsilon = epsilon.expand_as(latent_u)
		# epsilon = epsilon.cuda()

		# interpolation = epsilon*latent_u + (1 - epsilon)*latent_m

		# interpolation = Variable(interpolation, requires_grad=True)
		# interpolation = interpolation.cuda()

		# interpolation_logits = disc(interpolation)
		# grad_outputs = torch.ones(interpolation_logits.size())
		# grad_outputs = grad_outputs.cuda()

		# gradients = autograd.grad(outputs=interpolation_logits,
		# 						  inputs=interpolation,
		# 						  grad_outputs=grad_outputs,
		# 						  create_graph=True,
		# 						  retain_graph=True)[0]

		# gradients = gradients.view(batch_size, -1)
		# gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
		# grad_penalty = gamma * ((gradients_norm - 1) ** 2).mean()

		disc_out_u = disc(latent_u)
		disc_out_m = disc(latent_m)

		loss_disc = torch.mean(disc_out_m) - torch.mean(disc_out_u)# + grad_penalty
		loss_disc.backward(retain_graph = True)
		optim_dis.step()

		#Clip weights of discriminator
		for p in disc.parameters():
			p.data.clamp_(-0.05, 0.05)


		#Train generator every 5 iterations
		optim_gen.zero_grad()
		disc_fake_out = disc(latent_m)
		loss_adv = -torch.mean(disc_fake_out)
		loss_recn = criterion_recn(recon_m, imgs_m)
		# if(i % 3 == 0):
		loss_gen = loss_adv*0.1 + loss_recn*0.9
		# else:
			# loss_gen = loss_recn

		loss_gen.backward()
		optim_gen.step()

		disc_fake_out = disc_fake_out.cpu().detach().numpy()
		disc_fake_out = [1 if disc_fake_out[i][0] >= 0.5 else 0 for i in range(disc_fake_out.shape[0])]
		gen_accuracy = np.mean(disc_fake_out)

		disc_out_m = disc_out_m.cpu().detach().numpy()
		disc_out_u = disc_out_u.cpu().detach().numpy()

		disc_out_m = [1 if disc_out_m[i][0] >= 0.5 else 0 for i in range(disc_out_m.shape[0])]
		disc_out_u = [1 if disc_out_u[i][0] >= 0.5 else 0 for i in range(disc_out_u.shape[0])]

		disc_real_acc = np.mean(disc_out_u)
		disc_fake_acc = 1 - np.mean(disc_out_m)

		angle_error = torch.acos(cos(gaze_m, labels_m)).mean()*180.0/3.1415
		if(i % 1 == 0):
			print("epoch : {} iters : {} loss_recn : {:.3} loss_adv : {:.3} loss_disc : {:.3} angle_error : {:.3} disc_r_acc : {:.3} disc_f_acc : {:.3} gen_acc : {:.3} best : {:.3}".format(EPOCHS, i, loss_recn.item(), loss_adv.item(), loss_disc.item(), angle_error, disc_real_acc, disc_fake_acc, gen_accuracy, best_val_loss*1.0))
		# log_value('angle_error', angle_error, int(EPOCHS*(len(mpiloader)) + i))
		# log_value('loss_recn', loss_recn.item(), int(EPOCHS*(len(mpiloader)) + i))
		# log_value('loss_adv', loss_adv.item(), int(EPOCHS*(len(mpiloader)) + i))
		# log_value('loss_disc', loss_disc.item(), int(EPOCHS*(len(mpiloader)) + i))
		if(i == 0):
			save_img(imgs_m.cpu(), recon_m.cpu(), i)

		i += 1
		cosine_sim_train += angle_error

		if(angle_error < best_val_loss):
			torch.save(ae.state_dict(), save_dir + 'best_val_ae.wts')
			torch.save(disc.state_dict(), save_dir + 'best_val_disc.wts')
			best_val_loss = angle_error_test
			print("Saved best loss weights")

		if(i % 100 == 0):
			torch.save(ae.state_dict(), save_dir + 'trained_ae.wts')
			torch.save(disc.state_dict(), save_dir + 'trained_disc.wts')



		# angle_error_test = 0
		# if(i % 10 == 0):
		# 	for (imgs, labels) in mpiloader_test:
		# 		recon, gaze, latent = ae(imgs)
		# 		angle_error_test += torch.acos(cos(gaze, labels)).mean()*180.0/3.1415

		# 	angle_error_test /= len(mpiloader_test)
		# 	print("Validation Error : {:.3f}".format(angle_error_test))
		# 	if(angle_error_test < best_val_loss):
		# 		torch.save(ae.state_dict(), save_dir + 'best_val_ae.wts')
		# 		torch.save(disc.state_dict(), save_dir + 'best_val_disc.wts')
		# 		best_val_loss = angle_error_test
		# 		print("Saved best loss weights")
		# 	torch.save(ae.state_dict(), save_dir + 'trained_ae.wts')
		# 	torch.save(disc.state_dict(), save_dir + 'trained_disc.wts')



	cosine_sim_train /= len_loader

	print("EPOCH : {} Cosine_Sim_Train : {:.3f} ".format(EPOCHS, cosine_sim_train))
	torch.save(ae.state_dict(), save_dir + 'trained_ae.wts')
	torch.save(disc.state_dict(), save_dir + 'trained_disc.wts')
	
