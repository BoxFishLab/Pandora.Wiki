#/usr/bin/env python 
#-*- coding: utf-8 -*-

'''
线性回归模型
'''
from .basicmodel import BasicModule  #基本模型
import torch as t
from torch import nn
from torch.autograd import Variable as V
from torch import optim

class LinearRegression(BasicModule):

	def __init__(self,data,max_item=100000,aphla=0.01):
		super(LinearRegression,self).__init__()
		self.DefaultParameter(data)
		self.aphla = aphla
		self.max_item = max_item

	#默认参数设置
	def DefaultParameter(self,data):
		data_size = data.size()
		self.wights = nn.Parameter(t.randn(data_size[1],1))
		self.b = nn.Parameter(t.randn(data_size[0],1))
	#前馈函数
	def forward(self,x):
		y = x.mm(self.wights) + self.b
		return y
	def fit(self,lr,x,y,flag=t.cuda.is_available()):
		if flag:
			model = lr.cuda()
		else:
			model = lr
			loss_func = nn.MSELoss()
			optimizer = optim.SGD(params=model.parameters(),lr=model.aphla)
			for epoch in range(model.max_item):
				if flag:
					x = V(x).cuda()
					y = V(y).cuda()
				else:
					x = V(x)
					y = V(y)
				y_out = model(x)
				loss = loss_func(y_out,y)
				break_loss = loss.data[0]
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				if (epoch+1)%20 == 0 :
					items = (epoch+1)/20
					print("Epoch{0}次,代价函数为loss:{1:.6f}".format(items,loss.data))
				#满足一定条件就实现早停
				if break_loss < 1e-3:
					break
			model.save(model)
	def predict(self,lr,x,path):
		'''
		还需要做预测处理
		'''
		lr.eval()
		lr.load(path)
		y = lr(x)
		print(y)


