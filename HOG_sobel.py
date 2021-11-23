import cv2
import numpy as np
import matplotlib.pyplot as pl

hog= cv2.imread('/home/roopesh/Desktop/MediaPipe /result_images/result_2.jpg')
hog= np.float32(hog)/255.0

hog_x= cv2.Sobel(hog, cv2.CV_32F, 1, 0, ksize=1)
hog_y= cv2.Sobel(hog, cv2.CV_32F, 0, 1, ksize=1)

print('opencv version :', cv2.__version__)
hog1= hog_x+hog_y

# to calculate the magnitude and direction (in degrees)

mag, angle= cv2.cartToPolar(hog_x,hog_y, angleInDegrees=True)

hog2= mag+angle
cv2.imshow('',hog)
cv2.imshow('',hog1)


mag, angle
key = cv2.waitKey(1000000)
