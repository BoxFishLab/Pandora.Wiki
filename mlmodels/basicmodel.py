#!/usr/bin/env python
#-*- coding: utf-8 -*-

import torch as t
import time

'''
封装t.nn.module完成模型参数训练的保存和读取
	1.load(path)
	2.save(name)
'''
class BasicModule(t.nn.Module):
	def __init__(self,**args):
		'''
		默认model_name = modules.模型名
		'''
		super(BasicModule,self).__init__()
		self.modle_name = str(self.__class__.__name__)
	def load(self,path):
		'''加载模型有两种方式对应于保存模型的方式:
		1. 加载完整的模型结构和参数信息使用 load_model = torch.load('model. pth' ) ，在网络较大的时候加载的时间比较长，同时存储空间也比较大;
		2.(2)加载模型参数信息，需要先导人模型的结构，然后通过 model.load_state_dict(torch.load('model state.pth')) 来导入
		'''
		self.load_state_dict(t.load(path))
	def save(self,model,name=None):
		'''
		在 PyTorch 里面使用 torch.save 来保存模型的结构和参数，有两种保存方式:
		1. 保存整个模型的结构信息和参数信息，保存的对象是模型 model;torch.save(model , '.jmodel.pth ' )
		2. 保存模型的参数，保存的对象是模型的状态 model.state_dict(),torch.save(model.state_dict() f '. j model_state.pth') 
		'''
		if name is None:
			prefix = 'checkpoints/'+self.modle_name+'_'
			name = time.strftime(prefix+'%m%d_%H_%M_%S.pth')
		t.save(model.state_dict(),name)
		return name