#/usr/bin/env python
#-*- coding: utf-8 -*-

from torchvision.transforms import ToTensor as ToT
from torchvision.transforms import ToPILImage as ToPIL
from PIL import Image
import matplotlib.pyplot as plt

	def train_vision(self,x):
		x_size = x[0].shape
		print(x_size)
		channels = x
		for feature_graph in channels:
			print((feature_graph.unsqueeze(0)).shape)
			channels_pil = ToPIL()(feature_graph).convert('RGB')
			plt.imshow(channels_pil) 
			plt.show()
	#损失函数

	# #示例
	# image = Image.open("D:/Charben/天方夜谭_The days we spent/photos/mine/liuwei.jpg").convert('RGB')
	# image_tensor = ToT()(image).unsqueeze(0)
	# dl_out = dl_model(image_tensor)
	# print(dl_out)