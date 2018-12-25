#!/usr/bin/env python
#-*- coding: utf-8 -*-

import torchvision as tv
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import ToTensor as ToT
from torchvision.transforms import ToPILImage as ToPIL
from PIL import Image
import matplotlib.pyplot as plt


class LoadData():

	def __init__(self):
		self.transform = T.Compose([
			T.Resize(454),
			T.CenterCrop(454),
			T.ToTensor(),
			T.Normalize((.5,.5,.5),(.5,.5,.5))]
		)
	def load_data(self,root):
		train_data = tv.datasets.CIFAR10(
			root = root,
			train = True,
			download = True,
			transform = self.transform
		)
		train_loader = data.DataLoader(
			train_data,
			batch_size=4,
			shuffle=True,
			num_workers=2
			)
		test_data = tv.datasets.CIFAR10(
			root = root,
			train = False,
			download = True,
			transform = self.transform
		)
		test_loader = data.DataLoader(
			test_data,
			batch_size=4,
			shuffle=True,
			num_workers=2
			)
		classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
		# print("train_data的类型为：{0}，一共有{1}张".format(type(train_data),len(train_data)))
		# (image_data,image_label) = train_data[100]
		# print(image_data,classes[image_label])
		# image_data = (image_data + 1)/ 2
		return train_data,test_data,classes

def train_vision(image,label):
	channels_pil = ToPIL()(image).convert('RGB')
	plt.title(label)
	plt.imshow(channels_pil) 
	plt.show()

def test():
	root = "F:/MLWILL/Pandora.Liu/data/cifar_10/"
	load_data = LoadData()
	train_data,test_data,labels = load_data.load_data(root)
	(image_100,label_100) = train_data[500]
	image_100 = ((image_100+1)/2)
	train_vision(image_100,labels[label_100])

test()