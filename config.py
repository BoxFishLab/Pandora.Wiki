#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
配置程序config.py
	1.数据集参数：
	2.模型参数
	3.训练参数
	4.文件路径
	5.使用模型类型
	6.是否使用GPU
'''
class DefautConfig(object):

	#file position
	train_root = "D:/Charben/_Datasets/dogcat/train_mini/"
	test_root = "D:/Charben/_Datasets/dogcat/test/"
	model_save_root = ""
	debug_info = ""
	#hyper parameters
	split_rate = 0.75
	batch_size = 5
	num_workers = 2
	weight_decay = 1e-3
	lr = 0.01
	lr_rate = 0.95
	max_item = 100
	#Boolean
	use_gpu = True
	train = True
	test = True

	def parse(self,**kwargs):
		'''
		INfo: Class
		change the parameters
		'''
		for k,v in kwargs.items():
			print("k：{0}	v:{1}".format(k,v))
			if hasattr(k,self):
				setattr(k,v)
			else:
				print("请检查配置值")
def test():
	df = DefautConfig()
	parameters = {"lr":1e-3,"max_item":1000}
	df.parse(**parameters)

if __name__ == "__main__":
	test()