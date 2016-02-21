import matplotlib.pyplot as plt
import numpy as np


# theta_J = np.array([[-7.62078508,-6.07797498, 3.45854087],
#                      [ 6.41639135, 7.61186529, 2.89412162],
#                      [-3.11578858, 2.61127931, 0.26077626]])
# theta_J = np.array([[ 8.36375104, 6.81152197,-2.90731941],
#                      [-5.20700531, 6.6233405,  8.14867879],
#                      [ 1.41866317,-1.25519265,-0.7799916 ]])
# theta_K = np.array([[ 12.75741783],
#                      [-12.18250767],
#                      [  6.25787456]])

theta_J = np.array([[ 8.30783206, 6.88747479,-2.57799283],
                     [-5.24990104, 6.7286029, 8.0803377 ],
                     [ 1.49909855,-1.39486048,-1.08465038]])

theta_K = np.array([[-10.60278998],
                     [16.10302528],
                     [-10.58577542]])


'''plot lines for theta_J'''
# x2 = 1    # after 2nd, rest is bias-like for next layer
# countx0x1 = 0
# for i in range(theta_J.shape[1]):
#   w = theta_J[:,i]
#   x1 = np.linspace(-1,2,20)
#   x0 = -1  * (w[1] * x1 + w[2] * x2) / w[0]
#   plt.plot(x0,x1,'b-')

#   if(countx0x1<2):  # fist 2 lines special
#     plt.plot(x0,x1,'r-')
#   else:             # after 2nd, rest is bias-like for next layer
#     plt.plot(x0,x1,'b-')
#   countx0x1 += 1

'''plot lines for theta_K'''
# get one set
# extend to 4?...
# extend to line?....

x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
# print xx
# print yy
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
print z.shape
# plt.contourf(x,y,z)
# plt.show()


'''plot data sets'''
# a_I = np.array([[0,0],[0,1],[1,0],[1,1]])
# t_K = np.array([[0],[1],[1],[0]])

# x0 = a_I[:,0]
# x1 = a_I[:,1]
# y = t_K[:,0]

# plt.scatter(x0, x1, c=y, s=100, edgecolors='None')
# plt.colorbar()
# plt.grid()
# plt.show()

