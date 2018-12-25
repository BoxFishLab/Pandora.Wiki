#!/usr/bin/env python
#-*- coding: utf-8- -*-

'''
nms非极大值抑制算法：
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

dets = np.array([
	[55,40,188,188,0.9],
	[96,13,177,185,0.4],
	[101,52,202,180,0.85],
	[54,102,196,197,0.7],
	])
thresh = 0.3

def nms(dets,thresh):
	x1 = dets[:,0]
	y1 = dets[:,1]
	x2 = dets[:,2]
	y2 = dets[:,3]
	scores = dets[:,-1]
	areas = (x2-x1+1)*(y2-y1+1)
	order = scores.argsort()[::-1]#order = sorted(scores,reverse=True)
	keeps = [] #用于保存最终获得的bounding-box
	while order.size>0:
		i = order[0]
		keeps.append(i)
		xx1 = np.maximum(x1[i],x1[order[1:]])
		yy1 = np.maximum(y1[i],y1[order[1:]])
		xx2 = np.minimum(x2[i],x2[order[1:]])
		yy2 = np.minimum(y2[i],y2[order[1:]])
		w = np.maximum(0.0,xx2-xx1+1)
		h = np.maximum(0.0,yy2-yy1+1)
		inter = w*h
		ovr = inter/areas[i]+areas[order[1:]]-inter
		inds = np.where(ovr<thresh)[0]
		order = order[inds+1]
	return keeps
def test():
	img = mpimg.imread("D:/Charben/_Datasets/dogcat/train_mini/cat.1.jpg")
	fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,6))
	ax.add_patch(patches.Rectangle((0.1,0.1),0.5,0.3,fill=False,linewidth=10,edgecolor='r'))	
	ax.imshow(img)
	plt.show()

test()
bounding_box = nms(dets,thresh)
print(bounding_box)



