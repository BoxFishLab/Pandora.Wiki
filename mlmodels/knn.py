#!/usr/bin/env python
#-*- coding: utf-8 -*- 

'''
knn算法
'''
import numpy as np
import helper


class knn():
    
    #knn算法分类器
    def classify(self,inX,dataSet,labels,k):
        dataSetsize = dataSet.shape[0]
        print(dataSetsize)
        diffMat = tile(inX,(dataSetsize,1)) - dataSet #生成同纬度的计算向量
        sqdiffMat = power(diffMat,2)
        sqDistances = sqdiffMat.sum(axis=1)
        distances = power(sqDistances,0.5)
        sortDistances = distances.argsort()
        print(diffMat,sqDistances,distances,sortDistances)
        classcount = {}
        for i in range(k):
            label = labels[sortDistances[i]]
            classcount[label] = classcount.get(label,0)+1
        sortclass = sorted(classcount)
        sortclasssize = sortclass.__len__()
        if k>sortclasssize: k = sortclasssize
        for i in range(k):
            print("第",i+1,"类的数据类别分别为：",sortclass[i],"类有",classcount[sortclass[i]],"次,与目标距离最接近的为",distances[sortDistances[i]])
        return sortclass
    "可视化数据集"
    def plotdata(self):
        pass

path = "dataset.txt"
helper = helper.dataOperator()
dataMat,labelsMat = helper.readfile(path,"\t",3)  #
datavis = helper.dataVisual(object)
datavis.dataplot(dataMat,labelsMat)
