import cv2
import numpy as np


def image_difference(img1, img2):
    """
    Calculate the difference between two images.
    :param img1: First image
    :param img2: Second image
    :return: Difference image(img1 - img2)

    """
    diff= cv2.absdiff(img1,img2)
    mask= cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

    th=30
    imask =mask>th

    canvas =np.zeros_like(img2, np.uint8)
    canvas[imask]=img2[imask]
    cv2.imshow("img1",img1)
    cv2.imshow('img2',img2)
    cv2.imshow('result', canvas)
    cv2.waitKey(100000)

    return canvas


if __name__ == '__main__':
    pass
    _1 = cv2.imread('/home/roopesh/Desktop/MediaPipe /result_images/image109.png')
    _2 = cv2.imread('/home/roopesh/Desktop/MediaPipe /result_images/image109mask.jpg')
    image_difference(_1,_2)
