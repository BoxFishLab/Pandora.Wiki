#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
Vgg模型
''' 

from .basicmodel import BasicModule
import torch as t
from torch import optim
from torch import nn
from torch.autograd import Variable as V
class Vgg(BasicModule):

	def __init__(self):
		super(Vgg,self).__init__()
		self.layer1 = t.nn.Sequential(
			t.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True),			
			t.nn.ReLU(inplace=True),
			)
		self.layer2 = t.nn.Sequential(
			t.nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
			t.nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),			
			t.nn.ReLU(inplace=True),
			)
		self.layer3 = t.nn.Sequential(
			t.nn.MaxPool2d(kernel_size=2,stride=2),
			t.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			)
		self.layer4 = t.nn.Sequential(
			t.nn.MaxPool2d(kernel_size=2,stride=2),
			t.nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			)
		self.layer5 = t.nn.Sequential(
			t.nn.MaxPool2d(kernel_size=2,stride=2),
			t.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
			t.nn.ReLU(inplace=True),
			t.nn.MaxPool2d(kernel_size=2,stride=2),
			)
		self.layer6 = t.nn.Sequential(
			t.nn.Linear(25088,4096),
			t.nn.ReLU(inplace=True),
			t.nn.Linear(4096,4096),
			t.nn.ReLU(inplace=True),
			t.nn.Linear(4096,2),
			t.nn.ReLU(inplace=True),
			)
	def forward(self,x):
		x = self.layer1(x)
		#print(x.shape)
		x = self.layer2(x)
		#print(x.shape)
		x = self.layer3(x)
		#print(x.shape)
		x = self.layer4(x)
		#print(x.shape)
		x = self.layer5(x)
		#print(x.shape)
		x = x.view(x.size(0),-1)
		x = t.nn.functional.log_softmax(self.layer6(x),dim=1)
		#print(x.shape)
		# x = self.layer7(x)
		# print(x.shape)
		return x
	#目标函数和训练器	
	def fit(self,Vgg,train_data_loader,validation_data_loader,max_item=1000,lr=0.01,mu=0.95):
		'''
		随机梯度下降法进行训练:
		parameters:
			lr: mu*lr
			max_item: max_item
		'''
		self.loss = []
		self.validation_data_loader_size = len(validation_data_loader)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(params=Vgg.parameters(),lr=lr)
		for epoch in range(max_item):
			for batch_data,batch_label in train_data_loader:
				for (data,label) in zip(batch_data,batch_label): 
					data = V(data)
					optimizer.zero_grad()
					output = self.forward(data)
					target = V(t.LongTensor([int(label)]))
					loss = criterion(output,target)
					break_loss = loss.item()
					loss.backward()
					optimizer.step()
					#lr = mu*lr
					if break_loss < 1e-3:
						break
					self.loss.append(break_loss)
				'''
				模型验证Step
				'''
				error = 0
				Vgg.eval() #修改模型的验证
				for vali_data,vali_label in validation_data_loader:
					vali_data = V(vali_data.squeeze(0))
					target = V(t.LongTensor([int(vali_label[0])]))
					output = (self.forward(vali_data)).max(1,keepdim=True)[1]
					error += 1 if target!=output else 0
					error_rate = 100*error/self.validation_data_loader_size
				self.help(epoch,self.loss,error_rate)
				Vgg.train() #训练过程
		Vgg.save(Vgg)
		'''
		训练效果特别糟糕！！！
		'''
		return self
	def help(self,epoch,loss,accurity):
		print("第{0}次的损失loss为:{1},错误率为：{2:.4f}%".format(epoch+1,loss[epoch],accurity))