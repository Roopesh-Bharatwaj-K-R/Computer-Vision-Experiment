
import cv2
from pathlib import Path

import cv_exp.basic
import cv_exp.basic as cv
import cv_exp.pupil_detection as pupil_detection
import cv_exp.draw as draw
import cv_exp.log as log
import cv_exp.take as take
import matplotlib.pyplot as plt
import numpy as np
import glob


from cv_exp._cpp.pupil_detection import _segment_iris_with_iris_and_eyelids
from cv_exp._cpp.common_eyes import _DetectedPupils



model_face_detection = '/home/roopesh/Desktop/MediaPipe /cv-exp-framework-master/data/models/face_detection_front.tflite'
model_face_landmarks ='/home/roopesh/Desktop/MediaPipe /cv-exp-framework-master/data/models/face_landmark.tflite'
model_iris_landmarks = '/home/roopesh/Desktop/MediaPipe /cv-exp-framework-master/data/models/iris_landmark.tflite'
take_folder=('/home/roopesh/Desktop/MediaPipe /cv-exp-framework-master/images/image3.png')

iris_detector = pupil_detection.IrisDetectorMP(model_face_detection_path=model_face_detection,
                                                   model_face_landmarks_path=model_face_landmarks,
                                                   model_iris_landmarks_path=model_iris_landmarks)

#globbing utility.
import glob

def path_to_images(path):
    # path=('/home/roopesh/Desktop/MediaPipe /cv-exp-framework-master/images/*.png')
    for file in glob.glob(path):
        images = cv2.imread(file)

# conversion numpy array into rgb image to show

        image_bgr_rgb = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        image_rgb_bgr = cv2.cvtColor(image_bgr_rgb, cv2.COLOR_RGB2BGR)

        print('Image Shape',image_rgb_bgr.shape)
        out = image_rgb_bgr.copy()
        out_pupile = image_rgb_bgr.copy()
        detected_pupils, result = iris_detector.detect(image_rgb_bgr)

        if detected_pupils.is_ok():
            out_pupiles = draw.draw_pupils(out_pupile, detected_pupils.left, detected_pupils.right)
            cv2.imwrite('/home/roopesh/Desktop/MediaPipe /result_images/out_pupiles.jpg', out_pupiles)
            # cv2.imshow('Pupil detection', out_pupiles)
            # cv2.imshow('detected pupils', detected_pupils)
            print('coord of pupils left , right :', (detected_pupils.left, detected_pupils.right))

        if result is not None:
            (face_detection, face_keypoints, face_landmarks,
                left_iris_landmarks, left_eyelid_landmarks,
                right_iris_landmarks, right_eyelid_landmarks,
                left_iris_segmentation, right_iris_segmentation,
                concat_landmarks) = result
            out = draw.draw_contour(out, left_iris_segmentation, thickness=1, color=(0, 255, 0))
            out = draw.draw_contour(out, right_iris_segmentation, thickness=1, color=(0, 255, 0))
            # cv2.imshow('segment detection', out)
            # print('coord of eye segment :', (left_iris_segmentation, right_iris_segmentation))
            print('X,Y Points of Pupil :', (left_iris_landmarks[0], right_iris_landmarks[0]))


#  IMAGE SUBTRACTION







#  Extract specific pixels from image using CV2.THRESHOLD

            gray_out_pupiles=cv2.cvtColor(out_pupiles, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Pupil detection', gray_out_pupiles)

            retval, thresh_gray = cv2.threshold(gray_out_pupiles, thresh=17, maxval=255, type=cv2.THRESH_BINARY)
            print("threshlod",thresh_gray.shape)
            print("retval",retval)
            #
            points = np.argwhere(thresh_gray == 0)
            filter_points = np.fliplr(points)
            x, y, w, h = cv2.boundingRect(filter_points)
            x, y, w, h = x-10, y-10, w+20, h+20
            crop = gray_out_pupiles[y:y+h, x:x+w]

            retval, threshold_crop = cv2.threshold(crop, thresh=17, maxval=255, type=cv2.THRESH_BINARY)

            cv2.imshow('crop and thresh image', threshold_crop)
            threshold_crop_color= cv2.cvtColor(threshold_crop, cv2.COLOR_GRAY2BGR)
            threshold_crop_color_rgb=cv2.cvtColor(threshold_crop, cv2.COLOR_BGR2RGB)
            cv2.imshow('crop and thresh image_ new',    threshold_crop_color_rgb)


            # cv2.imshow('out_pupiles', out_pupiles)
            k = cv2.waitKey(100000)
            if k == 27:
                cv2.destroyAllWindows()





if __name__ == '__main__':
    p= ('/home/roopesh/Desktop/MediaPipe /cv-exp-framework-master/images/*.png')
    path_to_images(p)
