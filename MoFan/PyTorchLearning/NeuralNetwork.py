



from tkinter import NE
import numpy as np

#定义激活函数sigmoid ：f(x) = 1 / ( 1 + e^(-x))
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

#定义神经网络
class Neuron :
    #定义初始化函数
    def __init__(self,weights,bias) :
        self.weights = weights
        self.bias = bias
    
    def forward(self,inputs) :
        total = np.dot(self.weights,inputs) + self.bias
        return sigmoid(total)

weights = np.array([0,1])  # 定义权重 w1 = 0, w2 = 1
bias = 4
n = Neuron(weights,bias)

x = np.array([2,3])
print(n.forward(x)) 

class NeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
  '''

  def __init__(self) -> None:
    weights = np.array([0,1])
    bias = 0

    self.h1 = Neuron(weights,bias)
    self.h2 = Neuron(weights,bias)
    self.o1 = Neuron(weights,bias)
    

  def forward(self,x):
      out_h1 = self.h1.forward(x)
      out_h2 = self.h2.forward(x)

      #o1的输入是h1与h2的输出
      out_o1 = self.o1.forward(np.array(out_h1,out_h2))

      return out_o1

network = NeuralNetwork()
x = np.array([2,3])
print(network.forward(x))