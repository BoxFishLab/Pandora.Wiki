#!/usr/bin/env python
#-*- coding： utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor as ToT
from torchvision.transforms import ToPILImage as ToPIL

class Visualize(object):
	def __init__(self):
		pass
	def vis_online(self,neural_layer):
		'''
		卷积神经网络在线学习视图
		'''
		pass
	def vis_result(self,images,result):
		'''
		image: 图片路径
		result: 经过Conv卷积nn给出的结果
		'''
		size = len(images)
		fig,aix = plt.subplots(figsize=(6,8))
		fig.suptitle("Results of this picture")
		plt.ion()
		for i in range(size):
			image = images[i]
			img = mpimg.imread(image)
			aix.text(3,32,'The Conv_NN gives result:',style='italic',bbox={'facecolor':'red','alpha':0.5,'pad':10})
			aix.text(18,32,result,fontsize=15)
			aix.imshow(img)
		plt.show(True)
		plt.cla()
		plt.pause(0.033)
		pass
	def vis_analysis(self):
		'''
		评估指标：
		loss: 损失函数
		acc_rate:正确率
		others
		'''
		pass
	def train_vision(self,x):
		x_size = x[0].shape
		print(x_size)
		channels = x
		for feature_graph in channels:
			print((feature_graph.unsqueeze(0)).shape)
			channels_pil = ToPIL()(feature_graph).convert('RGB')
			plt.imshow(channels_pil) 
			plt.show()
def test():
	vis = Visualize()
	images = ["F:/MLWILL/Pandora.Liu/data/mnist/test_folder/0.0_1.jpg","F:/MLWILL/Pandora.Liu/data/mnist/test_folder/8.0_37095.jpg"]
	vis.vis_result(images,8)
if __name__ == '__main__':
	test()
