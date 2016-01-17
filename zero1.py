''' Online learning; layers of theta defined by case. '''
'''a = activation'''
'''I = input'''
'''J,K are random layer labels'''

import numpy as np
def logistic(x):
  return 1/(1+np.exp(-x))

import random
import generateData
training_data = generateData.make_OR_Dataset()
E = []

a_I, t_K = random.choice(training_data) #print 'a_I, t_K',a_I, t_K

I_width = a_I.shape[0]  # print 'I_width',I_width
J_width = 2
K_width = t_K.shape[0] # print K_width

theta_J = np.random.random((J_width,I_width))*2 - 1
theta_K = np.random.random((K_width,J_width))*2 - 1 # print 'theta_J,theta_K',theta_J,theta_K

a_J = 1/(1+np.exp((-1)*np.dot(theta_J,a_I))) # print 'a_J.shape, a_J',a_J.shape, a_J
a_K = 1/(1+np.exp((-1)*np.dot(theta_K,a_J))) # print 'a_K.shape, a_K',a_K.shape, a_K

# pattern: whenever new layer interact, transpose takes place..
dumpy_K = (t_K - a_K)*a_K*(1-a_K)
gradient_K = np.dot(dumpy_K,a_J.T)
dumpy_J = np.dot(dumpy_K,theta_K)*(a_J*(1-a_J)).T
gradient_J = np.dot(dumpy_J.T,a_I.T)

for i in range(10000):
  a_I, t_K = random.choice(training_data)
  a_J = 1/(1+np.exp((-1)*np.dot(theta_J,a_I)))
  a_K = 1/(1+np.exp((-1)*np.dot(theta_K,a_J)))
  E.append(np.sum((t_K - a_K)**2)/2)
  dumpy_K = (t_K - a_K)*a_K*(1-a_K)
  gradient_K = np.dot(dumpy_K,a_J.T)
  dumpy_J = np.dot(dumpy_K,theta_K)*(a_J*(1-a_J)).T
  gradient_J = np.dot(dumpy_J.T,a_I.T)
  theta_J = theta_J + gradient_J
  theta_K = theta_K + gradient_K

import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()



