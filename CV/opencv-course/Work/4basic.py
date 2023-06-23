import cv2 

img = cv2.imread("./Resources/Photos/park.jpg")
cv2.imshow('Park', img) 

# converting to grayscale image 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Gray', gray) 

# Blurring an image with Gaussian Kernel of size 3
blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
cv2.imshow('Blurred', blur)

# Edge Detection 
canny = cv2.Canny(img, 125, 175) 
cv2.imshow('Canny Edges', canny)

# We can reduce the number of edges by blurring it first 
canny_blur = cv2.Canny(blur, 125, 175) 
cv2.imshow('Canny Blur Edges', canny_blur)

# Dilating the image 
dilated = cv2.dilate(canny_blur, (7, 7), iterations=3) 
cv2.imshow("Dilated", dilated)

# eroding an image 
eroded = cv2.erode(dilated, (3, 3), iterations=1) 
cv2.imshow("Eroded", eroded )

# Resize an image 
resized = cv2.resize(img, (500, 500)) 
cv2.imshow("Resized", resized)

# Crop an image 
cropped = img[50:200, 200:400] 
cv2.imshow('Cropped', cropped)

cv2.waitKey(0)