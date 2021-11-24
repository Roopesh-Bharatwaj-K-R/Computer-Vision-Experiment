
import cv2
import numpy as np

original_image = cv2.imread('/home/roopesh/Desktop/MediaPipe /result_images/image109.png')
duplicate_image = cv2.imread('/home/roopesh/Desktop/MediaPipe /result_images/image109mask.jpg')

# check if 2 images equal or not
if np.array_equal(original_image, duplicate_image):
    print('np.array_equal : Images are equal', 'org :', np.array_equal(original_image), 'dup :', np.array_equal(duplicate_image))
else:
    print('np.array_equal : Images are not equal', 'org :', np.array(original_image), 'dup :', np.array(duplicate_image))

# check if 2 images equal or not
if original_image.shape == duplicate_image.shape:
    print('image.shape: Images are equal', 'org :',original_image.shape, 'dup:', duplicate_image.shape)
else:
    print('image.shape: Images are not equal')


# check if 2 images equal or not
if original_image.dtype == duplicate_image.dtype:
    print('image.dtype: Images are equal', 'org :',original_image.dtype, 'dup:', duplicate_image.dtype)
else:
    print('image.dtype:Images are not equal')

differences=cv2.subtract(original_image, duplicate_image)
b, g,r=cv2.split(differences)

if cv2.countNonZero(b)==0 and cv2.countNonZero(g)==0 and cv2.countNonZero(r)==0:
    print('Images are equal')
else:
    print('Images are not equal as it has mask')

cv2.imshow('original_image',original_image)
cv2.imshow('duplicate_image',duplicate_image)
cv2.imshow('differences',differences)
k = cv2.waitKey(100000)
if k == 27:
    cv2.destroyAllWindows()



