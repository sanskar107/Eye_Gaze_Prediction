from scipy.io import loadmat
from scipy.misc import imresize
import numpy as np
import os, sys
import cv2
import collections
import torch
import torchvision
import scipy.misc as m
import matplotlib.pyplot as plt
from skimage import io
from torch.utils import data
from glob import glob
import json
import random

class mpiloader(data.Dataset):
	def __init__(self, root, split="train", is_transform=False, img_size=None, augmentations=None):
		self.root = root
		self.split = split
		self.img_size = [35, 55]
		self.mean = 108.9174
		self.stddev = 48.6903

		prefix = 'MPIIGaze/Evaluation Subset/sample list for eye image/'
		prefix1 = 'MPIIGaze/Data/Normalized/'
		files = os.listdir(prefix)
		path = []
		for f in files:
			path += [prefix1 + f.split('.')[0] + '/' + i for i in open(prefix + f).read().split('\n')[:-1]]
		self.files = path

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		path = self.files[index]
		mat = loadmat(path.split(' ')[0][:-9] + '.mat')
		img = mat['data'][0][0][1 if path.split(' ')[1] == 'right' else 0][0][0][1][int(path.split(' ')[0].split('/')[-1][:-4]) - 1]
		img = imresize(img[1:, 3:-2], (35, 55))

		lbl = mat['data'][0][0][1 if path.split(' ')[1] == 'right' else 0][0][0][0][int(path.split(' ')[0].split('/')[-1][:-4]) - 1]

		img = img.astype(np.float64)
		img = (img - self.mean)/self.stddev

		img = torch.from_numpy(img.copy()).float().unsqueeze(0)
		lbl = torch.from_numpy(lbl).float()

		return img, lbl


if __name__ == '__main__':

	loader = mpiloader('./all/', split="train")
	
	trainloader = data.DataLoader(loader, batch_size = 1)
	
	for i, data in enumerate(trainloader):
		imgs, labels = data
		print(imgs.shape, labels.shape)
		break
