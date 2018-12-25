#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
图片预处理程序
'''
from PIL import Image 
import os
from torch.utils import data
from torchvision import transforms as T
class LoadImage(data.Dataset):
	def __init__(self,root,rate=0.7,transform=None,train=True,test=False):
		self.root =root
		self.image = self.load_image()
		division_size = int(rate*len(self.image))
		if test and (train is False):
			self.image = self.image
		if train:
			self.image = self.image[:division_size]
		else:
			self.image = self.image[division_size:]
		if transform is None:
			self.transforms = T.Compose([
			T.Resize(224),
			T.CenterCrop(224),
			T.ToTensor(),
			T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
			])
		else:
			self.transforms = transform
	def __getitem__(self,index):
		img_path = self.image[index]
		label =1 if (img_path.split("/")[-1]).split(".")[0] == 'cat' else 0
		image = Image.open(img_path)
		if self.transforms:
			#label_data = T.ToTensor(label)
			image_data = self.transforms(image)
			image_data = image_data.unsqueeze(0)
			return image_data,label
	def __len__(self):
		return len(self.image)
	def data_loader(self,data_train,batch_size=5,shuffle=True,num_workers=2):
		return data.DataLoader(data_train,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
	def load_image(self):
		self.image = []
		suffix = [".bgm",".jpeg",".gif",".psd",".png",".jpg"]
		if os.path.exists(self.root) != True:
			print("文件不存在，请检查路径是否拼写错误！")
		try:
			listpath =  os.listdir(self.root)
			for filedir in listpath:
				currfile = self.root + filedir
				if os.path.isfile(currfile):  #判断该文件是不是一个文件
					piclen = len(suffix)
					for i in range(piclen):
						if currfile.endswith(suffix[i]):
							self.image.append(currfile)
				else:
					currfile += "/"
					self.load_image(currfile)
		except Exception as e:
			print(str(e))
		return self.image