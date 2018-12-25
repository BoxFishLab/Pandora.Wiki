#!/usr/bin/env python
#-*- coding: utf-8 -*-

from .basicmodel import BasicModule  #基本模型
from torch import nn
from torch import optim

class Yolo(BasicModule):
	def __init__(self):
		super(Yolo,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=0,dilation=1,bias=True),
			nn.ReLU(inplace=True)
		)
		self.layer2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2,stride=2)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True)
		)
		self.layer4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2,stride=2)
		)
		self.layer5 = nn.Sequential(
			nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer6 = nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True)
		)
		self.layer7 = nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0,dilation=1,bias=True),
			nn.ReLU(inplace=True)
		)
		self.layer8 = nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True)
		)
		self.layer9 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2,stride=2)
		)
		self.layer10 = nn.Sequential(
			nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer11 = nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True)
		)
		self.layer12 = nn.Sequential(
			nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0,dilation=1,bias=True),
			nn.ReLU(inplace=True)
		)
		self.layer13 = nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer14 = nn.Sequential(
			nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer15 = nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer16 = nn.Sequential(
			nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0,dilation=1,bias=True),
			nn.ReLU(inplace=True)
		)
		self.layer17 = nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer18 = nn.Sequential(
			nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer19= nn.Sequential(
			nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer20 = nn.Sequential(
			nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True)
		)
		self.layer21 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2,stride=2)
		)
		self.layer22 = nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,stride=1),
			nn.ReLU(inplace=True)
		)
		self.layer23 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2,stride=2)
		)
		self.layer24 = nn.Sequential(
			nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,stride=1)
		)
		self.fc = nn.Sequential(
			nn.Linear(25600,4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096,1470),
			nn.ReLU(inplace=True)
		) 
	def forward(self,x):
		print("原始尺寸：",x.shape)
		x = self.layer1(x)
		print(x.shape)
		x = self.layer2(x)
		print(x.shape)
		x = self.layer3(x)
		print(x.shape)
		x = self.layer4(x)
		print(x.shape)
		x = self.layer5(x)
		print(x.shape)
		x = self.layer6(x)
		print(x.shape)
		x = self.layer7(x)
		print(x.shape)
		x = self.layer8(x)
		print(x.shape)
		x = self.layer9(x)
		print(x.shape)
		x = self.layer10(x)
		print(x.shape)		
		x = self.layer11(x)
		print(x.shape)
		x = self.layer12(x)
		print(x.shape)
		x = self.layer13(x)
		print(x.shape)
		x = self.layer14(x)
		print(x.shape)
		x = self.layer15(x)
		print(x.shape)		
		x = self.layer16(x)
		print(x.shape)
		x = self.layer17(x)
		print(x.shape)
		x = self.layer18(x)
		print(x.shape)
		x = self.layer19(x)
		print(x.shape)
		x = self.layer20(x)
		print(x.shape)		
		x = self.layer21(x)
		print(x.shape)
		x = self.layer22(x)
		print(x.shape)
		x = self.layer23(x)
		print(x.shape)
		x = self.layer24(x)
		print(x.shape)
		x = x.view(x.size(0),-1)
		x = self.fc(x)
		print(x.shape)
		x = x.view(7,7,30)
		print(x.shape)
		return x