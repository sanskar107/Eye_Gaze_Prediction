import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from skimage import io
from torch.utils import data
from glob import glob
import json

class eyeloader(data.Dataset):
	def __init__(self, root, split="train", is_transform=False, img_size=None, augmentations=None):
		self.root = root
		self.split = split
		self.img_size = [35, 55]
		self.augmentations = augmentations

		if split == "train":
			self.files = glob(root + 'imgs_cropped/*.png')[0:800]
		elif split == "val":
			self.files = glob(root + 'imgs_cropped/*.png')[800:]
			

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		img_name = self.files[index]
		img_path = img_name
		lbl_path = img_path.replace('_cropped.png', '.json')

		img = io.imread(img_path)
		img = np.array(img, dtype=np.uint8)

		lbl = list(map(float, json.loads(open(lbl_path).read())["eye_details"]["look_vec"][1:-1].split(',')))
		lbl = np.array(lbl[:3]).astype(np.float32)
		lbl[1] = -lbl[1]
		
		img = img.astype(np.float64)

		img = torch.from_numpy(img).float().unsqueeze(0)
		lbl = torch.from_numpy(lbl).float()

		return img, lbl

if __name__ == '__main__':

	loader = eyeloader('./', split="val")
	
	trainloader = data.DataLoader(loader, batch_size = 1)
	
	for i, data in enumerate(trainloader):
		imgs, labels = data

		print(imgs.shape, labels.shape)
		print(labels)
		break
