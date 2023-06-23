import cv2 

# A color space is a multidimensional representation of the color specturm. 
# RGB : 3D space (red green blue) all 0-255
# HSV : (hue saturation value), like a cylndrical representation, with polar angle (hue 0-360), saturation (0-100%), value (0-100) 
# HSV generalizes how humans perceive color, so it is most accurate depiction of how we feel colors on screen. 
# LAB : Another color space

img = cv2.imread('Resources/Photos/park.jpg')
cv2.imshow('Park', img) 

# BGR/RGB to Grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow("Grayscale", gray) 

# BGR to HSV 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
cv2.imshow("HSV", hsv)

# BGR to LAB color space 
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB", lab) 

# Turns out that BGR is an inverse version of RGB 
import matplotlib.pyplot as plt 
# plt.imshow(img) # inverted color scheme 

# BGR to RGB 
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
cv2.imshow("RGB", rgb) 
plt.imshow(rgb) 
plt.show() 


cv2.waitKey(0)