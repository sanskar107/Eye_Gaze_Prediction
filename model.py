import torch
import torch.nn as nn
from torch.autograd import Variable

class AutoEncoder(nn.Module):

	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1, bias=True),
			nn.LeakyReLU(0.1, inplace=True),

			nn.MaxPool2d(kernel_size=2, stride=2),
			
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(0, 1), bias=True),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.LeakyReLU(0.1, inplace=True),

			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
			nn.LeakyReLU(0.1, inplace=True),
			# nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(512, 1024, kernel_size=7, stride=1, padding=0, bias=True),
			nn.LeakyReLU(0.1, inplace=True),
		   )
			# deconv
		self.decoder = nn.Sequential(

			nn.ConvTranspose2d(1024, 512, (3, 5)),
			nn.LeakyReLU(0.1, inplace=True),

			nn.ConvTranspose2d(512, 256, (3, 5)),
			nn.LeakyReLU(0.1, inplace=True),

			nn.ConvTranspose2d(256, 128, (3, 5)),
			nn.LeakyReLU(0.1, inplace=True),
			nn.ConvTranspose2d(128, 64, (3, 5)),
			nn.LeakyReLU(0.1, inplace=True),

			nn.ConvTranspose2d(64, 32, (5, 9)),
			nn.LeakyReLU(0.1, inplace=True),

			nn.ConvTranspose2d(32, 16, (5, 9)),
			nn.LeakyReLU(0.1, inplace=True),
			nn.ConvTranspose2d(16, 8, (7, 9)),
			nn.LeakyReLU(0.1, inplace=True),

			nn.ConvTranspose2d(8, 4, (7, 9)),
			nn.LeakyReLU(0.1, inplace=True),
			nn.ConvTranspose2d(4, 2, (7, 7)),
			nn.LeakyReLU(0.1, inplace=True),

			nn.ConvTranspose2d(2, 1, 1),
			nn.LeakyReLU(0.1, inplace=True),
		)

		self.gaze = nn.Sequential(
			nn.Linear(1024, 256),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Linear(256, 64),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Linear(64, 3),
			)


	def forward(self, x):
		latent = self.encoder(x)
		eye_gaze = self.gaze(latent.view(-1, 1024))
		recons = self.decoder(latent)
		return recons, eye_gaze, latent.view(-1, 1024)


class Descriminator(nn.Module):
	def __init__(self):
		super(Descriminator, self).__init__()
		self.disc = nn.Sequential(
			nn.Linear(1024, 256),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Linear(256, 64),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Linear(64, 1),
			nn.Sigmoid(),
			)
	def forward(self, x):
		return self.disc(x)


class RMSELoss(nn.Module):
	def __init__(self):
		super(RMSELoss).__init__()
		self.mse = nn.MSELoss()
		
	def forward(self,yhat,y):
		return self.mse(yhat,y)/10000

class CosineLoss(nn.Module):
	def __init__(self):
		super(CosineLoss).__init__()
		cos = nn.CosineSimilarity(dim = 1)
		
	def forward(self,yhat,y):
		return torch.acos(cos(yhat, y))


if __name__ == '__main__':
	# ae = AutoEncoder()
	# a = torch.randn(1, 1, 35, 55)
	# for params in ae.gaze.parameters():
	# 	params.requires_grad = False
	# for params in ae.parameters():
	# 	print(params)
	dis = Descriminator()
	a = torch.randn(1, 1024)
	print(dis(a))
