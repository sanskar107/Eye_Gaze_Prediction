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

		prefix = self.root + 'MPIIGaze/Evaluation Subset/sample list for eye image/'
		prefix1 = self.root + 'MPIIGaze/Data/Normalized/'
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


class unityloader(data.Dataset):
	def __init__(self, root, split="train", is_transform=False, img_size=None, augmentations=None):
		self.root = root
		self.split = split
		self.img_size = [35, 55]
		self.augmentations = augmentations
		self.mean = 108.9174
		self.stddev = 48.6903

		self.files = open(self.root + self.split + '.txt', 'r').read().split('\n')[:-1]

		# if split == "train":
		# 	# self.files = [root + 'imgs_cropped/' + str(i) + '_cropped.png' for i in range(1, 800)]
		# 	self.files = glob(root + 'imgs_cropped/*.png')[0:800]
		# elif split == "val":
		# 	# self.files = [root + 'imgs_cropped/' + str(i) + '_cropped.png' for i in range(800, 1000)]
		# 	self.files = glob(root + 'imgs_cropped/*.png')[800:]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		img_name = self.files[index]
		img_path = self.root + 'imgs_cropped/' + img_name
		lbl_path = img_path.replace('_cropped.png', '.json').replace('imgs_cropped', 'imgs')


		img = io.imread(img_path)
		img = np.array(img, dtype=np.uint8)

		img = (img - img.min())*(255.0/(img.max() - img.min()))
		lbl = list(map(float, json.loads(open(lbl_path).read())["eye_details"]["look_vec"][1:-1].split(',')))
		lbl = np.array(lbl[:3]).astype(np.float32)
		lbl[1] = -lbl[1]
		
		img = img.astype(np.float64)
		img = (img - self.mean)/self.stddev
		if(self.split == 'train'):
			if(random.random() > 0.5):
				# img = img[:, ::-1]
				img = np.fliplr(img)
				lbl[0] = -1*lbl[0]

		img = torch.from_numpy(img.copy()).float().unsqueeze(0)
		lbl = torch.from_numpy(lbl).float()

		return img, lbl

if __name__ == '__main__':

	loader = unityloader('./all/', split="train")
	loader1 = mpiloader('./', split = 'train')
	unityloader = data.DataLoader(loader, batch_size = 1)
	mpiloader = data.DataLoader(loader1, batch_size = 1)

	for i, (a, b) in enumerate(zip(unityloader, mpiloader)):
		c, d = a
		print(c.shape, d.shape)
		# d_u, d_m = data
		# img, lab = d_u
		# print(img.shape, lab.shape)
		# img, lab = d_m
		# print(img.shape, lab.shape)
		break