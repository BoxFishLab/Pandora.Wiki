#！/usr/bin/env python
#-*- coding: utf-8 -*-
'''
第一个现代卷积神经网络模型：LeNet-5
'''
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable as V 
from .basicmodel import BasicModule  #基本模型
class LeNet(BasicModule):
	'''	
	pytorch中卷积神经网络模块都在nn中。
	其二维卷积函数为:nn.Conv2d(**kwargs):
		**kwargs参数设置：{
			in_channels: 输入通道数,
			out_channels:输出通道数,
			kernel_size: 卷积核尺寸,
			stride： 步长,
			padding: 零填充,
			bias: 偏置（True/False）,默认表示使用偏置
		}
	其二维最大值池化函数为: MaxPool2d(**kwargs):
		**kwargs参数设置：{
			kernel_size: 卷积核尺寸,
			stride： 步长,
			padding: 零填充,
			return_indices:表示是否返回最大值所处的下标
		}
	其二维平均池化函数为：nn.AvgPool2d(**kwargs)表示均值池化
	还有部分参数详见官网
	'''
	def __init__(self,**args):
		super(LeNet,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0,dilation=1,bias=True),
			nn.BatchNorm2d(6),
			nn.ReLU(inplace=True)
			)
		self.layer2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2,stride=2)
			) 
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0,dilation=1,bias=True),
			nn.BatchNorm2d(16), #规范化层
			nn.ReLU(inplace=True)
			)
		self.layer4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2,stride=2)
			)
		self.layer5 = nn.Sequential(
			nn.Conv2d(in_channels=16,out_channels=120,kernel_size=3,stride=1,padding=0,dilation=1,bias=True),
			nn.BatchNorm2d(120),
			nn.ReLU(inplace=True)
			)
		#Dropout层，应该提供，防止过拟合
		self.fc = nn.Sequential(
			nn.Linear(480,84),
			nn.ReLU(inplace=True),
			nn.Linear(84,10),
			nn.ReLU(inplace=True),
			) 
	#前向传播
	def forward(self,x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = x.view(x.size(0),-1)
		x = self.fc(x)
		return x
	#目标函数和训练器	
	def fit(self,LeNet,train_data_loader,validation_data_loader,max_item=1000,lr=0.01,mu=0.95):
		'''
		随机梯度下降法进行训练:
		parameters:
			lr: mu*lr
			max_item: max_item
		'''
		self.loss = []
		self.validation_data_loader_size = len(validation_data_loader)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(params=LeNet.parameters(),lr=lr)
		for epoch in range(max_item):
			for batch_data,batch_label in train_data_loader:
				for (data,label) in zip(batch_data,batch_label): 
					data = V(data)
					optimizer.zero_grad()
					output = self.forward(data)
					target = V(torch.LongTensor([int(label)]))
					loss = criterion(output,target)
					break_loss = loss.item()
					loss.backward()
					optimizer.step()
					#lr = mu*lr
					if break_loss < 1e-3:
						break
					self.loss.append(break_loss)
				error = 0
				'''
				模型验证Step
				'''
				LeNet.eval() #修改模型的验证
				for vali_data,vali_label in validation_data_loader:
					vali_data = V(vali_data.squeeze(0))
					target = V(torch.LongTensor([int(vali_label[0])]))
					output = (self.forward(vali_data)).max(1,keepdim=True)[1]
					error += 1 if target!=output else 0
					accurity = 100*error/self.validation_data_loader_size
					print("Accurity:",accurity)
				self.help(epoch,self.loss,accurity)
				LeNet.train() #训练过程
		LeNet.save(LeNet)
		'''
		训练效果特别糟糕！！！
		'''
		return self
	def help(self,epoch,loss,accurity):
		print("第{0}次的损失loss为:{1},错误率为：{2:.4f}%".format(epoch+1,loss[epoch],accurity))
	def predict(self,x):
		pass
