#!/us/bin/env python
#-*- coding: utf-8 -*-

'''
idx_ubyte文件解析:以及保存.jpg图片
'''

import numpy as np
import struct
from PIL import Image 

class IdxUbyte(object):

	def __init__(self):
		self.images = np.zeros((60000,28,28))
		self.label = np.empty(60000)
	def decode_idx3_ubyte(self,root):
		bin_data = open(root,'rb').read()
		offset = 0
		fmt_header = '>iiii'
		magic_number,images_num,rows_num,cols_num = struct.unpack_from(fmt_header,bin_data,offset)
		image_size = rows_num * cols_num
		offset += struct.calcsize(fmt_header)
		fmt_image = '>' + str(image_size) + 'B'
		for i in range(images_num):
			print("正在解析：{}张".format(i+1))
			self.images[i] = (np.mat(np.array(struct.unpack_from(fmt_image,bin_data,offset)).reshape((rows_num,cols_num))))
			offset += struct.calcsize(fmt_image)
		return self.images
	def decode_idx1_ubyte(self,root):
		bin_data = open(root,'rb').read()
		offset = 0
		fmt_header = '>ii'
		magic_number,images_num= struct.unpack_from(fmt_header,bin_data,offset)
		offset += struct.calcsize(fmt_header)
		fmt_image = '>B'
		for i in range(images_num):
			print("正在解析：{}张".format(i+1))
			self.label[i] = (np.mat(np.array(struct.unpack_from(fmt_image,bin_data,offset))[0]))
			offset += struct.calcsize(fmt_image)
		return self.label
	def save_image_jpg(self,images,labels,folder):
		size = len(images)
		for i in range(size):
			image_ipg = Image.fromarray(images[i],mode='RGB')
			images_jpg_file = folder+str(i)+".jpg"
			image_ipg.save(images_jpg_file)
			print("已保存{}张".format(i))
def test():
	#train_images_idx3_ubyte_file = "F:/MLWILL/Pandora.Liu/data/mnist/train-images.idx3-ubyte"
	#train_images_idx1_ubyte_file = "F:/MLWILL/Pandora.Liu/data/mnist/train-labels.idx1-ubyte"
	test_images_idx3_ubyte_file = "F:/MLWILL/Pandora.Liu/data/mnist/t10k-images.idx3-ubyte"
	test_images_idx1_ubyte_file = "F:/MLWILL/Pandora.Liu/data/mnist/t10k-labels.idx1-ubyte"
	#train_folder = "F:/MLWILL/Pandora.Liu/data/mnist/train_folder/"
	test_folder = "F:/MLWILL/Pandora.Liu/data/mnist/test_folder/"
	idx_ubyte = IdxUbyte()
	#train_images = idx_ubyte.decode_idx3_ubyte(train_images_idx3_ubyte_file)
	#train_labels = idx_ubyte.decode_idx1_ubyte(train_images_idx1_ubyte_file)
	test_images = idx_ubyte.decode_idx3_ubyte(test_images_idx3_ubyte_file)
	test_labels = idx_ubyte.decode_idx1_ubyte(test_images_idx1_ubyte_file)
	#idx_ubyte.save_image_jpg(train_images,train_labels,train_folder)
	idx_ubyte.save_image_jpg(test_images,test_labels,test_folder)


if __name__ == '__main__':
	test()

