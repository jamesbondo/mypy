'''How should one tweak the input slightly to increase the output?'''
    

        '''Strategy #1: Random Local Search'''

import numpy as np
def forwardMultiplyGate(x,y):
  return x*y
x = -2
y = 3

tweak_amount = 0.01
best_out = -100
best_x = x
best_y = y
for k in range(0,100):
  print k
  x_try = x + tweak_amount * (np.random.random() * 2 - 1)
  y_try = y + tweak_amount * (np.random.random() * 2 - 1)
  out = forwardMultiplyGate(x_try,y_try)
  if(out > best_out):
    best_out = out
    best_x = x_try
    best_y = y_try
print best_out,best_x,best_y