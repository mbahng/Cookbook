import cv2 

img = cv2.imread('/home/mbahng/Desktop/Cookbook/CV/opencv-course/Resources/Photos/park.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

print(type(img), img.shape) 
print(type(gray), gray.shape) 


cv2.imshow("Park", img[100:200, 100:200, :])     # Show RGB image
# cv2.imshow('Gray', gray)    # Show gray image
cv2.waitKey(0)              # Wait time until image closes