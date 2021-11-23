
import cv2
import numpy as np

# Load image, create mask, and draw white circle on mask
image = cv2.imread('/home/roopesh/Desktop/MediaPipe /cv-exp-framework-master/TAKE_FOLDER/1525.37329237500011913653_rgb.png')

print(image.shape)
mask = np.zeros(image.shape, dtype=np.uint8)
# start_point=((388, 854))
# end_point=((428,824))
# mask= cv2.rectangle(mask, start_point,end_point, (255,255,255), -1)
mask = cv2.circle(mask, (194, 412), 225, (255,255,255), -1)

# Mask input image with binary mask
result = cv2.bitwise_and(image, mask)
# Color background white
result[mask==0] = 255 # Optional

cv2.imshow('image', image)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()






