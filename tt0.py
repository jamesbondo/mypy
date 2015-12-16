import numpy as np
from numpy import linspace,array,random
import matplotlib.pyplot as plt


def create_X_y(m,n):
  ones = np.ones(m)
  x = np.random.random((m,n))
  count = 0
  while (count < n):
    x[:,count] += np.arange(m)*np.random.random()*np.random.randint(10)
    count = count + 1
  X = np.c_[ones,x]
  y = np.linspace(0,5,m) + np.random.random(m)
  return X,y

def scale_std(matrix):
  m, n = X.shape
  count = 1
  while (count<n):
    array = matrix[:,count]
    matrix[:,count] = (array - array.mean())/array.std()
    count = count +1
  return matrix

m = 4
n = 2
X,y = create_X_y(m,n)
X = scale_std(X)
alpha = 0.01
J = []
theta = np.random.random(n+1)  # print theta.shape #(4,)

for i in range(100):
  hypothesis = np.dot(X,theta)
  diff = hypothesis - y
  J.append(np.sum(diff**2)/(2*m))
  gradients = np.dot(diff,X)/m
  theta = theta - alpha * gradients
# print theta
plt.plot(J,'o')
plt.show()
