import numpy as np 
import cv2 

# Masking allows us to focus on a specific region of an image
img = cv2.imread('/home/mbahng/Desktop/Cookbook/CV/opencv-course/Resources/Photos/cats.jpg')
cv2.imshow("Cats", img) 

blank = np.zeros(img.shape[:2], dtype='uint8')
cv2.imshow('Blank Image', blank) 

mask = cv2.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1) 
cv2.imshow('Mask', mask) 

masked = cv2.bitwise_and(img, img, mask=mask) 
cv2.imshow('Masked Image', masked)

cv2.waitKey(0)