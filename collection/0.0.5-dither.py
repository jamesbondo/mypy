========================== threshold dither ======================
'''black/white '''
'''vectorized, pretty cool'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('images/ghost.1.0.png',0)

safer = 0.0000001
threshhold = 0.05   # extreme values to discover eyes! magical
imgMax = np.amax(img[0:5])
imgMax = np.float32(imgMax)
img = (img/(imgMax+safer) >= threshhold)*1
# plt.imshow(img)
plt.imshow(img,'gray')

plt.show()



