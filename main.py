#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
主程序
'''
from config import DefautConfig as Cfg
from data import LoadImage
from dlmodels import Vgg
from dlmodels import Yolo
import torch as t

def test_dogvscat():
	train_li = LoadImage(Cfg.train_root,train=True,rate=Cfg.split_rate)
	vali_li = LoadImage(Cfg.train_root,train=False,rate=Cfg.split_rate)
	print("一共有{0}张图片参与训练,{1}张图片参与验证".format(len(train_li),len(vali_li)))
	#test_li = LoadImage(test_root,train=False,test=True)
	cfg = Cfg()
	train_li_loader = train_li.data_loader(train_li,batch_size=Cfg.batch_size,shuffle=True,num_workers=Cfg.num_workers)
	vali_li_loader = vali_li.data_loader(vali_li,batch_size=1,shuffle=True,num_workers=Cfg.num_workers)
	vgg =Vgg()
	vgg.fit(vgg,train_li_loader,vali_li_loader,max_item=cfg.max_item,lr=cfg.lr,mu=Cfg.lr_rate)

if __name__ == '__main__':
	test_dogvscat()
