# # import numpy as np

# # def logistic(x):
# #   return 1/(1+np.exp(-x))
# # def logisticDERIV(x):
# #   return x*(1-x)

# # x = np.array([[0,0],[0,1],[1,0],[1,1]])
# # y = np.array([0,1,1,0])
# # ones = np.ones(x.shape[0])
# # X = np.c_[ones,x]
# # alpha = 0.01
# # m, n = X.shape
# # '''# only one neuron in receiving layer'''
# # theta = np.random.random(n)

# # for i in range(1000):
# #   hypothesis = logistic(np.dot(X,theta))
# #   diff = hypothesis - y
# #   gradients = np.dot(diff,X)/m
# #   theta = theta - alpha*gradients

# # print theta




import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel

def logistic(x):
  return 1/(1+np.exp(-x))
def logisticDERIV(x):
  return x*(1-x)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
ones = np.ones(x.shape[0])
X = np.c_[ones,x]
alpha = 0.01
m, n = X.shape
J=[0]
theta = np.random.random(n)

for i in range(1000):
  hypothesis = logistic(np.dot(X,theta))
  diff = hypothesis - y
  # gradients = np.dot(diff,X)/m
  # theta = theta - alpha*gradients

  delta = diff * logisticDERIV(hypothesis)
  theta += np.dot(X.T,delta)

  # y1 = (-1)*(np.dot(y,np.log(hypothesis+0.00000001)))
  # y0 = np.dot(y-1,np.log(1-hypothesis + 0.00000001))
  # Ji = (y1 + y0)/m
  # J.append(Ji)

print logistic(np.dot(X,theta))
# plt.plot(J,'o')
# plt.show()


#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


#  in order to "understand" language, it's necessary to define a new language that will be updated 
# constantly; just like a computer language.

# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
# from random import choice 
# from numpy import array, dot, random

# unit_step = lambda x: 0 if x < 0 else 1
# training_data = [ (array([0,0,1]), 0), 
#                 (array([0,1,1]), 1), 
#                 (array([1,0,1]), 1), 
#                 (array([1,1,1]), 1), 
#                 ]
# w = random.rand(3)
# errors = []
# eta = 0.2
# n = 100
# for i in xrange(n):
#   x, expected = choice(training_data)
#   result = dot(w, x)
#   error = expected - unit_step(result)
#   errors.append(error)
#   w += eta * error * x

# for x, _ in training_data:
#   result = dot(x, w)
#   print("{}: {} -> {}".format(x[:2], result, unit_step(result)))  


# # http://glowingpython.blogspot.com/2011/10/perceptron.html
# from pylab import rand,plot,show,norm

# class Perceptron:
#  def __init__(self):
#   """ perceptron initialization """
#   self.w = rand(2)*2-1 # weights
#   self.learningRate = 0.1

#  def response(self,x):
#   """ perceptron output """
#   y = x[0]*self.w[0]+x[1]*self.w[1] # dot product between w and x
#   if y >= 0:
#    return 1
#   else:
#    return -1

#  def updateWeights(self,x,iterError):
#   """
#    updates the weights status, w at time t+1 is
#        w(t+1) = w(t) + learningRate*(d-r)*x
#    where d is desired output and r the perceptron response
#    iterError is (d-r)
#   """
#   self.w[0] += self.learningRate*iterError*x[0]
#   self.w[1] += self.learningRate*iterError*x[1]

#  def train(self,data):
#   """ 
#    trains all the vector in data.
#    Every vector in data must have three elements,
#    the third element (x[2]) must be the label (desired output)
#   """
#   learned = False
#   iteration = 0
#   while not learned:
#    globalError = 0.0
#    for x in data: # for each sample
#     r = self.response(x)    
#     if x[2] != r: # if we have a wrong response
#      iterError = x[2] - r # desired response - actual response
#      self.updateWeights(x,iterError)
#      globalError += abs(iterError)
#    iteration += 1
#    if globalError == 0.0 or iteration >= 100: # stop criteria
#     print 'iterations',iteration
#     learned = True # stop learning
# def generateData(n):
#  """ 
#   generates a 2D linearly separable dataset with n samples. 
#   The third element of the sample is the label
#  """
#  xb = (rand(n)*2-1)/2-0.5
#  yb = (rand(n)*2-1)/2+0.5
#  xr = (rand(n)*2-1)/2+0.5
#  yr = (rand(n)*2-1)/2-0.5
#  inputs = []
#  for i in range(len(xb)):
#   inputs.append([xb[i],yb[i],1])
#   inputs.append([xr[i],yr[i],-1])
#  return inputs


# trainset = generateData(30) # train set generation
# perceptron = Perceptron()   # perceptron instance
# perceptron.train(trainset)  # training
# testset = generateData(20)  # test set generation

# # Perceptron test
# for x in testset:
#  r = perceptron.response(x)
#  if r != x[2]: # if the response is not correct
#   print 'error'
#  if r == 1:
#   plot(x[0],x[1],'ob')  
#  else:
#   plot(x[0],x[1],'or')

# # plot of the separation line.
# # The separation line is orthogonal to w
# n = norm(perceptron.w)
# ww = perceptron.w/n
# ww1 = [ww[1],-ww[0]]
# ww2 = [-ww[1],ww[0]]
# plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--k')
# show()