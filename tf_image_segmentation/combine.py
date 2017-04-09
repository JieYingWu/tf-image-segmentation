import matplotlib.pyplot as plt
import cv2
import numpy as np
from color_transfer import color_transfer
import compose

fg = cv2.imread('foreground2.png')
bg = cv2.imread('background.png')
mask = np.float32(cv2.imread('mask2.png'))

img = np.uint8(color_transfer(bg,fg))
#rows,cols = img[:,:,1].shape
#M = np.float32([[1,0,-100],[0,1,0]])
#img = cv2.warpAffine(img,M,(cols,rows))
#mask = cv2.warpAffine(mask,M,(cols,rows))

#Blur the mask outwards
kernel = np.ones((3,3), np.float32) / 9
dst = cv2.filter2D(mask, -1, kernel)
new_mask = dst / np.amax(dst)
new_mask[mask == 1] = 1
mask = new_mask

composed = np.uint8(compose.join(img,bg,mask))

cv2.imwrite('composed.png', composed)