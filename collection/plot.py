

======================''' simple drawing of the cost function result'''======================
import matplotlib.pyplot as plt
plt.plot(J,'o')
plt.show()

======================''' 3d drawing'''======================
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)

scale = 20
order = 4
X = np.arange(-scale*0.5, scale*0.5, 0.25)
Y = np.arange(-scale*0.5, scale*0.5, 0.25)
X, Y = np.meshgrid(X, Y)
# R = 5*(X)**2 + Y**2+X
# R = 5*(X)**2 + Y**2+X + X**3+Y**3
R = X*Y
Z = (R)/(scale*order)-scale/4*2
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-scale, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-scale, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=scale, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-scale, scale)
ax.set_ylabel('Y')
ax.set_ylim(-scale, scale)
ax.set_zlabel('Z')
ax.set_zlim(-scale, scale)

plt.show()
