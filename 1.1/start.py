
"""
start.py is procedural control. 
    currently also contain upper stream "neurons". 
"""
from random import choice 
from numpy import array, dot, random

import generateData
training_data = generateData.make_OR_Dataset()

import layer_w_1
layerw1 = layer_w_1.Layer_w_1()

import feedbackSYS
feedback = feedbackSYS.Feedback_1(layerw1.response)  
          # initialize with desired response function


errors = []
n = 1000
for i in xrange(n):
  x, feedback.expected = choice(training_data) 
  error = feedback.expected - layerw1.output(x)
  layerw1.updateWeights(x,error)
  errors.append(error) 



for x, _ in training_data: 
  result = dot(x, layerw1.w)
  print("{}: {} -> {}".format(x[:2], result, layerw1.output(x)))


# import matplotlib.pyplot as plt
# plt.plot(errors, marker='o')
# plt.title('Perceptron')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.show()