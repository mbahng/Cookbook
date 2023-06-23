import cv2 
import numpy as np 

# contours are not like edges, more used in segmentation

# We can find contours from looking at the edges first (let's blur first)
img = cv2.imread("/home/mbahng/Desktop/Cookbook/CV/opencv-course/Resources/Photos/cats.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
canny = cv2.Canny(blur, 125, 175)
cv2.imshow("Canny Edges", canny) 

# Take the edges and find the contours 
contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours found') 

# We can visualize this by drawing the contours over a blank image 
blank = np.zeros(img.shape, dtype='uint8') 
# Draw contours over image (-1: all contours, color red, thickness 2) 
cv2.drawContours(blank, contours, -1, (0, 0, 255), 2)
cv2.imshow("Contours Drawn", blank)


cv2.waitKey(0)


