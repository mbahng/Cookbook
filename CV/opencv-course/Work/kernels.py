import cv2 
import numpy as np 

img = cv2.imread("/home/mbahng/Desktop/Cookbook/CV/opencv-course/Resources/Photos/park.jpg")

kernel1 = np.array([[-1,  0,  1], [-2, 0, 2], [-1,  0,  1]])
kernel2 = np.array([[-1,  -2,  -1], [0, 0, 0], [1,  2, 1]])

img1 = cv2.filter2D(img, -1, kernel1)
img2 = cv2.filter2D(img, -1, kernel2)
# gaussian_blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)


# cv2.imshow('Gaussian_Blur', gaussian_blur)
cv2.imshow("Vertical", img1)
cv2.imshow("Horizontal", img2)
cv2.waitKey(0) 