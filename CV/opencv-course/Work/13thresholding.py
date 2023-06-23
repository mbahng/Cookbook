import cv2
import matplotlib.pyplot as plt

# Creating a binary image from a colored imaged based on whether pixel 
# intensity is above or below some value 

img = cv2.imread('/home/mbahng/Desktop/Cookbook/CV/opencv-course/Resources/Photos/cats.jpg')
cv2.imshow("Cats", img) 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Gray', gray)

# Simple thresholding 
threshold, thresh = cv2.threshold(src=gray,     
                                  thresh=150,   # threshold value
                                  maxval=255,   # max value
                                  type=cv2.THRESH_BINARY)   # convert to 0 or maxval
cv2.imshow('Simple Thresholded', thresh)

# Adaptive Thresholding 
# Doesn't require you to make your own threshold value 
# Computer finds the optimal value for you 

adaptive_thresh = cv2.adaptiveThreshold(src=gray, 
                                        maxValue=255, 
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        thresholdType=cv2.THRESH_BINARY, 
                                        blockSize=11, 
                                        C=3) # int value that is substracted from mean to fine tune 
cv2.imshow('Adaptive Thresholded', adaptive_thresh)

cv2.waitKey(0) 