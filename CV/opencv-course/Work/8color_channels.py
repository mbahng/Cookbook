import cv2 
import numpy as np 

img = cv2.imread('Resources/Photos/park.jpg')
cv2.imshow('Park', img) 

# splits the image into blue, green, red 
b, g, r = cv2.split(img) 

# Grayscale versions
# cv2.imshow('Blue', b) 
# cv2.imshow('Green', g) 
# cv2.imshow('Red', r) 

print(img.shape) # (height, width, 3) 
print(b.shape)   # (height, width) 
print(g.shape)   # (height, width) 
print(r.shape)   # (height, width) 

# Let's put the actual colors by defining the 3-tuple 
blank = np.zeros(img.shape[:2], dtype='uint8')
blue = cv2.merge([b, blank, blank]) 
green = cv2.merge([blank, g, blank]) 
red = cv2.merge([blank, blank, r]) 

cv2.imshow('Blue', blue) 
cv2.imshow('Green', green)
cv2.imshow('Red', red) 

# merge the color chanels together to get the original image 
merged = cv2.merge([b, g, r]) 
cv2.imshow('Merged Image', merged) 

# Generate random noise 
rand_red_img = np.random.randint(0, 255, size=(500, 300), dtype='uint8') 
rand_blu_img = np.random.randint(0, 255, size=(500, 300), dtype='uint8') 
rand_gre_img = np.random.randint(0, 255, size=(500, 300), dtype='uint8')

cv2.imshow("Red Noise", rand_red_img)
cv2.imshow("Blue Noise", rand_blu_img)
cv2.imshow("Green Noise", rand_gre_img)

merged = cv2.merge([rand_blu_img, rand_gre_img, rand_red_img])
cv2.imshow("Noise", merged)


cv2.waitKey(0) 