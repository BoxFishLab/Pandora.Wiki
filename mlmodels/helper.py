#!/usr/bin/env python
#-*- coding: utf-8 -*- 

'''
辅助程序：完成功能
'''
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

"数据预处理操作"
class dataOperator():

	"附带label s列的TXT文本文件,读取k列,返回类型为： matrix"
	def readfile(self,filename,delimiter,k):
		with open(filename) as file:
			content = file.readlines()
		size = len(content)
		dataSet = zeros((size,k-1))
		labelsSet = []
		index = 0
		for line in content:
			line = line.strip().split(delimiter)
			dataSet[index,:] = line[0:k-1]
			labelsSet.append(line[-1])
			index += 1
		dataMat = mat(dataSet)
		labelsMat = mat(labelsSet)
		return dataMat,labelsMat

	"数据归一化"
	def normdata(self,dataSet):
		#进一步处理数据，进行数据归一化
		min = dataSet.min(0)
		max = dataSet.max(0)
		ranges = max - min
		normdata = zeros(shape(dataSet))
		m = dataSet.shape[0]
		normdata = dataSet - tile(min, (m,1))
		normdata = normdata/tile(ranges, (m,1))
		return normdata,ranges,min,labelsSet
	
	"数据预处理操作"
	def datapre(self,dataSet):
		self.dataSet = dataSet
		n = shape(self.dataSet)[1]
		for i in range(n):
			meanVal = mean(self.dataSet[nonzero(~isnan(dataSet[:i].A))[0],i])
			dataSet[nonzero(isnan(self.dataSet[:,i].A)[0],i)] = meanVal
		datapreSet = self.dataSet
	    #return datapreSet

class dataVisual(object):

	def dataplot(self,dataSet,labelSet):
		m,n = shape(dataSet)
		target = labelSet
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(m):
			if target[i] == 1:
				ax.scatter(dataSet[i,0],dataSet[i,1],c='blue',marker='o')
			else:
				ax.scatter(dataSet[i,0],dataSet[i,1],c='red',marker='s')
		plt.show()		


class SimilarMean(object):


	"闵可夫斯距离,当P=1时为曼哈顿距离，p=2时为欧式距离，p>3时minkows"
	def minkowskidis(self,matA,matB,p):
		middle = matA-matB
		if p == 1:
			sum_middle = abs(power(middle,p).sum(axis=1))
			return power(sum_middle,1/p)
		sum_middle = power(middle,p).sum(axis=1)
		return power(sum_middle,1/p)
	
	"切比雪夫距离"
	def chebyshevdis(self,matA,matB):
		return (abs(matA-matB)).max()

	"夹角余弦距离"
	def cosine(self,matA,matB):
		return dot(matA,matB.T)/(linalg.norm(matA)*linalg.norm(matB))


