import cv2 
import numpy as np

img = cv2.imread("/home/mbahng/Desktop/Cookbook/CV/opencv-course/Resources/Photos/park.jpg")
cv2.imshow("Park", img) 

# Note that we have affine transformations in E(2), which is a subgroup of the 
# general linear group GL(R3)

# translation 
def translate(img, x, y): 
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0]) 
    
    return cv2.warpAffine(img, transMat, dimensions)

# Translate right by 100 pixels and down by 100 pixels 
translated = translate(img, 100, 100)
cv2.imshow("Translated", translated)

# Rotation around any point in img 
def rotate(img, angle, rotPoint=None): 
    (height, width) = img.shape[:2] 
    
    if rotPoint is None: 
        rotPoint = (width // 2, height // 2)
        
    # scale value is 1.0 since we don't want to scale it 
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0) 
    dimensions = (width, height) 
    
    return cv2.warpAffine(img, rotMat, dimensions) 

rotated = rotate(img, 45) 
cv2.imshow("Rotated", rotated)

# Resize an image (different ways to but cubic is highest quality) 
resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC) 
cv2.imshow("Resized", resized)

# Flipping (0, flip vert, 1: flip hori, -1: both vert and hori) )
flip = cv2.flip(img, -1) 
cv2.imshow("Flipped", flip)

cv2.waitKey(0) 