#!/usr/bin/env python
#-*- coding: utf-8 -*- 


"svd隐喻义算法"
class SVD():

    def __init__(self):
        self.testData = mat([[1,2,3,5,4],\
                            [2,5,7,2,4],\
                            [4,7,9,1,2],\
                            [9,1,7,4,2]])
        self.test = mat([[3,6,8,5,1]])
    def seleteR(self,trainSet,r):
        m,n = shape(trainSet)
        maxr = min(m,n) 
        if maxr<r: r = maxr  #对r的值进行选取?这里要进行判断
        U,S,VT = linalg.svd(trainSet)  #调用linalg.svd()计算奇异值
        sig = S**2
        ssig = sum(sig)*0.9    #计算总能量的90%
        ssig2 = sum(sig[:r])
        if ssig2 < ssig: 
            r += 1
            self.seleteR(trainSet,r)
        else:
            UR = U[:,:r]
            S = diag(S)
            SR = S[:r,:r]
            VTR = VT[:r,:]
            '''
            print(r,"个奇异值为：",SR)
            print("原矩阵为：",self.testData)
            print("原矩阵的相似结果：",UR*SR*VTR)
            '''