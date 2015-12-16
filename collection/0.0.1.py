# batch_gradient_descent_algo.py
import numpy as np
from sklearn.datasets.samples_generator import make_regression 
import pylab


def gradient_descent_2(alpha, x, y, numIterations):
    m = x.shape[0]
    theta = np.ones(2)
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        diff = hypothesis - y
        # J = np.sum(diff ** 2) / (2 * m)
        gradient = np.dot(x.T, diff) / m
        theta = theta - alpha * gradient  # update
    return theta

if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 
                        random_state=0, noise=35)
    m, n = x.shape
    x = np.c_[ np.ones(m), x] # insert column
    alpha = 0.01 # learning rate
    theta = gradient_descent_2(alpha, x, y, 1000)

    for i in range(x.shape[1]):
        y_predict = theta[0] + theta[1]*x
    pylab.plot(x[:,1],y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()
    print "Done!"


================


import numpy as np
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt
# x,y = make_regression(n_samples=500, n_features=1, n_informative=1,random_state=1, noise=35)
x = np.linspace(-5,5,50)
y = np.linspace(0,5,50) + np.array(np.random.random(50))

alpha = 0.01
m = x.shape[0]
ones = np.ones(x.shape[0])
J = []
X52 = np.c_[ones,x]
theta12 = np.random.random(X52.shape[1])

for i in range(1000):
  hypothesis = np.dot(X52,theta12.T)
  diff = hypothesis - y
  J.append(np.sum(diff**2)/(2*m))
  gradients = np.dot(diff, X52)/m
  theta12 = theta12 - alpha*gradients

plt.plot(x,y,'o')
plt.plot(x,np.dot(X52,theta12.T),'-')
# plt.plot(J,'o')
plt.show()


==================

import numpy as np
from numpy import linspace,array,random
import matplotlib.pyplot as plt
m = 50
n = 3
x = linspace(0,5,m)
randomX1 = x + array(random.random(m))
randomX2 = x + array(random.random(m))*5+10
randomX3 = x + array(random.random(m))*0.1-10
# plt.plot(x,randomX1,'o')
# plt.plot(x,randomX2,'x')
# plt.plot(x,randomX3,'-')
# plt.show()
y = np.linspace(0,5,50) + np.array(np.random.random(50))


ones = np.ones(m)
X = np.c_[ones, randomX1,randomX2,randomX3]
alpha = 0.0002
J = []
theta = np.random.random(n+1)  # print theta.shape #(4,)

for i in range(100):
  hypothesis = np.dot(X,theta)
  diff = hypothesis - y
  J.append(np.sum(diff**2)/(2*m))
  gradients = np.dot(diff,X)/m
  theta = theta - alpha * gradients


plt.plot(J,'o')
plt.show()

======================


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

m = 40
n = 3
X,y = create_X_y(m,n)
X = scale_std(X)
alpha = 0.02
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
