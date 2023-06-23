import cv2 
import numpy as np

# 2 types of edge detection: Laplacian and Sobel Gradient Magnitude Representation

img = cv2.imread('Resources/Photos/park.jpg')
cv2.imshow("Park", img) 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Gray', gray)

# Laplacian 
lap = cv2.Laplacian(gray, cv2.CV_64F) 
lap = np.uint8(np.absolute(lap)) 
cv2.imshow('Laplacian', lap) 

# Sobel 
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) 
combined_sobel = cv2.bitwise_or(sobelx, sobely) 

# cv2.imshow('Sobel X', sobelx) 
# cv2.imshow('Sobel Y', sobely)
cv2.imshow('Combined Sobel', combined_sobel) 


cv2.waitKey(0) 