
import pylab
import scipy.misc, scipy.optimize, scipy.io, scipy.special
from numpy import *
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

def displayData( X, theta = None ):
  width = 20
  rows, cols = 10, 10
  out = zeros(( width * rows, width*cols ))

  rand_indices = random.permutation( 5000 )[0:rows * cols]

  counter = 0
  for y in range(0, rows):
    for x in range(0, cols):
      start_x = x * width
      start_y = y * width
      out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
      counter += 1

  img   = scipy.misc.toimage( out )
  figure  = pyplot.figure()
  axes    = figure.add_subplot(111)
  axes.imshow( img )

  img   = scipy.misc.toimage( out )
  figure  = pyplot.figure()
  axes    = figure.add_subplot(111)
  axes.imshow( img )

  if theta is not None:
    result_matrix   = []
    X_biased    = c_[ ones( shape(X)[0] ), X ]
    
    for idx in rand_indices:
      result = (argmax( theta.T.dot(X_biased[idx]) ) + 1) % 10
      result_matrix.append( result )

    result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
    print result_matrix
  pyplot.show()



mat = scipy.io.loadmat( "./data/ex3data1.mat")
X, y      = mat['X'], mat['y']
displayData( X )


================