#!usr/bin/env python
#-*- coding:utf-8 -*-

'''
神经网络练习demo
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import json

#激活函数类
class Activate(object):
	#Sigmoid函数
	def Sigmoid(self,x):
		return 1/(1+np.exp(-x))
	#Sigmoid求导函数
	def DSigmoid(self,x):
		return self.Sigmoid(x)*(1-self.Sigmoid(x))
	#双曲正切函数
	def Tanh(self,x):
		return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	#双曲正切函数求导
	def DTanh(self,x):
		return 1-(self.Tanh(x))**2
	#硬限幅函数
	def Hardlim(self,x):
		y = np.where(x>=0,1,0)
		return y
	#硬限幅函数求导
	def DHardlim(self,x):
		return 0
	#斜面函数
	def Ramp(self,x):
		return np.where(np.abs(x)>=1,x,np.where(x<=-1,-1,1))
	#斜面函数求导
	def DRamp(self,x):
		return np.where(np.abs(x)>=1,1,0)	
	#修正线性单元
	def ReLU(self,x):
		return np.where(x>0,x,0)
	#修正线性单元求导函数
	def DReLU(self,x):
		return np.where(x>0,1,0)
	#渗透修正线性单元
	def LReLU(self,x,a=0.5 ):
		return np.where(x>=0,1,a*x)
	#渗透修正线性单元求导函数
	def DLReLU(self,x,a=5):
		return np.where(x>=0,1,a)
	#参数修正线性单元
	def PReLU(self,x,a=0.25):
		return np.where(x>=0,1,ax)
	#参数修正线性单元求导函数
	def DPReLU(self,x,a=0.25):
		return np.where(x>=0,1,a)
	#指数线性单元ELU
	def ELU(sef,x):
		return np.where(x>=0,1,a*(np.exp(x)-1))
	#指数线性单元ELU
	def DELU(self,x):
		return np.where(x>=0,0,-a*(np.exp(x))/(np.exp(x)-1)**2)
	#软加函数softmax
	def Softmax(self,x):
		return np.log(1+np.exp(x))
	#软加函数softmax的求导函数
	def DSoftmax(self,x):
		return np.exp(x)/(1+np.exp(x))
	def Maxout(self,x):
		return np.max(x)
	#软最大输出函数
	def SoftMaxout(self,x):
		exp_x = np.exp(x)
		exp_sum = exp_x.sum()
		return exp_x/exp_sum
	#软最大输出函数求导函数
	def DSoftMaxout(self,x):
		exp_x = np.exp(x)
		exp_sum = exp_x.sum(axis=1)
		return (2*exp_sum*exp_x)/(exp_sum)**2
# def test():
# 	act = Activate()
# 	x = np.linspace(-2,2,100)
# 	print(x)
# 	y_sigmoid = act.Sigmoid(x)
# 	y_tanh = act.Tanh(x)
# 	y_hardlim = act.Hardlim(x)
# 	y_relu = act.ReLU(x)
# 	y_softmaxout = act.SoftMaxout(x)
# 	y_lrelu = act.LReLU(x)
# 	fig = plt.figure()
# 	plt.title("Activate Function")
# 	plt.plot(x,y_sigmoid,color='red',lw=2,label="Sigmoid")
# 	plt.plot(x,y_tanh,color='yellow',lw=2,label="Tanh")
# 	plt.plot(x,y_hardlim,color='green',lw=2,label="Hardlim")
# 	plt.plot(x,y_relu,color='blue',lw=2,label="ReLU")
# 	plt.plot(x,y_softmaxout,color='pink',lw=2,label="SoftMaxout")
# 	plt.plot(x,y_lrelu,color='gray',lw=2,label="LReLU")
# 	plt.xlabel("-X-")
# 	plt.ylabel("-Y-")
# 	plt.legend()
# 	plt.grid(True)
# 	plt.show()
# if __name__ == '__main__':
# 	test()
"交叉熵代价函数"
class CrossEntropyCost(object):

	@staticmethod
	def Func(a,y):
		return np.sum(np.nan_to_num(-(y*np.log(a)+(1-y)*np.log(1-a))))

	@staticmethod
	def delta(a,y):
		return (a-y)
"二次代价函数"
class QuadraticCost(object):

	act = Activate()

	@staticmethod
	def Func(a,y):
		return (1/2)*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(a,y,z):
		return (a-y)*act.dsigmoid(z)


class NetWorks(object):

	act = Activate()

	'初始化神经网络结构'
	def __init__(self,neural,apha,maxIteration,mini_batch_size):
		self.num_layers = len(neural)   #神经网络的层次结构
		#self.DefaultWeightsBiases(neural)
		self.LargeWeightsBiases(neural)
		self.apha = apha
		self.maxIteration = maxIteration
		self.mini_batch_size = mini_batch_size
	"默认权值设置"
	def DefaultWeightsBiases(self,neural):
		self.biases = [np.random.randn(y,1) for y in neural[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(neural[:-1],neural[1:])] #一个列表存储着权值变化
	"采用高斯随机分布初始化"
	def LargeWeightsBiases(self,neural):
		self.biases = [np.random.randn(y,1) for y in neural[1:]]
		self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(neural[:-1],neural[1:])] #一个列表存储着权值变化


	"附带label s列的TXT文本文件,读取k列,返回类型为： matrix"
	def ReadFile(self,filename,delimiter,k,ratio=0.7):
		with open(filename) as file:
			content = file.readlines()
		size = len(content)
		dataSet = np.zeros((size,k))
		trainsize = int(ratio*size)         #选择数据集的70%作为训练集
		trainSet = np.zeros((trainsize,k))
		testsize = size - trainsize
		testSet = np.zeros((testsize,k))
		index = 0
		for line in content:
			line = line.strip().split(delimiter)
			if line[-1] == "Iris-setosa": line[-1] = 0
			elif line[-1] == "Iris-versicolor":line[-1] = 1
			else: line[-1] = 2
			dataSet[index,:] = line[0:k]
			index += 1
		dataMat = np.mat(dataSet)
		np.random.shuffle(dataMat)  #随机打乱数据集合
		trainSet[:trainsize,:k] = dataMat[:trainsize,:k]
		testSet[:testsize,:] = dataMat[trainsize:size,:k]
		return trainSet,testSet

	"随机梯度下降法"
	def SGD(self,trainData,testData):
		n = trainData.shape[0]
		mini_batches = []
		ratio_list = []
		result_list = []
		for i in range(self.maxIteration):
			np.random.shuffle(trainData)  #随机打乱数据
			for k in range(0,n,self.mini_batch_size):
				mini_batches.append(trainData[k:k+self.mini_batch_size])
			for mini_batche in mini_batches:
				for mini_xy in mini_batche:
					self.update_mini_batch(mini_xy)    #更新权值
			"代码跟踪"
			ratio = self.evaluate(testData,self.weights,self.biases)
			print("迭代{0}：；正确率为：{1:0.4f}%".format(i,ratio))
			ratio_list.append((i,self.weights,self.biases,ratio))
			result_list.append([i,ratio])
			#self.PlotRatio(x,y)
		return ratio_list,result_list
	"权值与阈值的更新"
	def update_mini_batch(self,mini_batche):
		self.nable_b = [np.zeros(b.shape) for b in self.biases]
		self.nable_w = [np.zeros(weight.shape) for weight in self.weights]
		mini_batche = np.mat(mini_batche)
		mini_data = mini_batche[:,:-1]     #数据测试列
		mini_labels = mini_batche[:,-1]    #数据标签列
		#print("单条数据输入X(mini_data):{0};单条数据输入x对应的标签Y(mini_labels):{1}".format(mini_data,mini_labels))
		for x,y in zip(mini_data,mini_labels):
			#self.backprogram(x,y)
			delta_nabla_b, delta_nabla_w = self.backprogram(x, y)
			self.nabla_b = [nb+dnb for nb, dnb in zip(self.nable_b, delta_nabla_b)] 
			self.nabla_w = [nw+dnw for nw, dnw in zip(self.nable_w, delta_nabla_w)] 
			self.weights = [w-(self.apha/self.mini_batch_size)*nw for w, nw in zip(self.weights, self.nabla_w)] 
			self.b = [b-(self.apha/self.mini_batch_size)*nb for b, nb in zip(self.biases, self.nabla_b)]

	"反向传播法"
	def backprogram(self,x,y):
		x = x.T
		y = y.astype("float")
		x = x.astype("float")
		activation = x
		activations = [x]   #储存Nerual network输出结果
		zs = []
		self.nable_b2 = [np.zeros(b.shape) for b in self.biases]
		self.nable_weights2 = [np.zeros(weight.shape) for weight in self.weights]
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation)+b
			zs.append(z)
			activation = self.act.sigmoid(z)
			activations.append(activation)
		#print("未变化输出神经元：\n{0}".format(activations[-1]))
		maxIndex = int(np.mat(activations[-1]).argmax(axis=0))
		error = int(y) - maxIndex  #"这里做了一个细节上思考的东西？分类结果对哪一类影响损失最大"
		#print("经变化输出三个神经元分别为y_put:\n{0},\n原标签ylabel=:\n{1},\n误差项为error_lavel：{2},\n总体误差error:\n{3}".format(activations[-1],y_label,error_lavel,error))
		delta = error*self.act.dsigmoid(zs[-1])
		self.nable_b2[-1] = delta
		self.nable_weights2[-1] = np.dot(delta,activations[-2].T) 
		#print("最外层偏置为:\n{0},\n权值为=:\n{1}".format(nable_b[-1],nable_weights[-1]))
		for i in range(2,self.num_layers):
			sb = self.act.dsigmoid(zs[-i])
			delta = np.dot(self.weights[-i+1].T,delta)  #反向梯度算法中的梯度求解？？？
			self.nable_b2[-i] = delta
			self.nable_weights2[-i] = np.dot(delta,activations[-i-1].T)
		#print("nable_b={0},nable_weights={1}".format(nable_b,nable_weights))
		return self.nable_b2,self.nable_weights2
		
	"计算计算值与真实值之间的差值"
	def costRF(self,R,F):
		return (R-F)

	"前馈信息输出"
	def feedforward(self,a,weights,biase):
		for b,w in zip(biase,weights):
			z = np.dot(w,a)+b
			a = self.act.sigmoid(z)
		return a  
	
	"使用测试集合检测权值偏置正确率"
	def evaluate(self,test_data,weights,biase):
		test_label = np.mat(test_data[:,-1:])
		test_data = np.mat(test_data[:,:-1])
		cost_sum = 0
		test_results = []
		for x,y in zip(test_data,test_label):
			finlly_output =self.feedforward(x.T,weights,biase)
			y = int(y)
			test_results.append([(np.argmax(finlly_output),y)])
		for test_result in test_results:
			for (x,y) in test_result:
				cost_sum += int(x==y)
		test_data_size = test_label.shape[0] 
		ratio = 100*(cost_sum/test_data_size)
		#print("神经网络最终得到的鸢尾花数据集正确率为：{0}%".format(ratio))
		return ratio

	"画出神经网络权值与阈值更新图"
	def PlotRatio(self,ratios):
		fig = plt.figure((10,3))
		fig.suptitle("Results")
		plt.plot(ratio[:,0].tolist(),ratio[:,1].tolist())
		plt.show()
'''
	"保存训练的权值"
	def SaveWeightsBiases(self,filepath):
		data = {
			"size":self.num_layers,
			"weights:":[weight.tolist() for weight in self.weights],
			"biases:":[biases.tolist() for biases in self.biases],
		}
		with open(filepath,'w+',encoding="utf-8") as file:
			json.dump(data,file)
'''
	
def test():
	path = "iris.txt"
	neural = [4,100,3]
	nn = NetWorks(neural,0.01,100,10)
	trainData,testData = nn.ReadFile(path,",",5,ratio=0.7) #trainData[-1]最后一列为标签
	ratio ,result= nn.SGD(trainData,testData)
	filepath = "D:/Charben/MLWLEE/FINALPROJECTS/神经网络/WeightsBiases.json"
	#nn.SaveWeightsBiases(filepath)
	
	#print("经过训练之后得到的权值weights：{0};\n偏置为b:{1}\n".format(nn.weights,nn.b))
	#b = nn.b
	#weights = nn.weights
	#nn.evaluate(testData,weights,b)
	nn.PlotRatio(result)
	"最优情况训练出99.666.%"
if __name__ =="__main__":
	test()


