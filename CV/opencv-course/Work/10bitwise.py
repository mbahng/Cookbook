import cv2 
import numpy as np 

blank = np.zeros((400, 400), dtype='uint8')

# draw a rectangle and circle 
rectangle = cv2.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1) 
circle = cv2.circle(blank.copy(), (200, 200), 200, 255, -1) 

cv2.imshow('Rectangle', rectangle) 
cv2.imshow('Circle', circle) 

# bitwise AND recturns intersection of these two images 
bitwise_and = cv2.bitwise_and(rectangle, circle) 
cv2.imshow('Bitwise AND', bitwise_and) 

# Bitwise OR returns union
bitwise_or = cv2.bitwise_or(rectangle, circle) 
cv2.imshow('Bitwise OR', bitwise_or) 

# Bitwise XOR returns non intersecting regions
bitwise_xor = cv2.bitwise_xor(rectangle, circle) 
cv2.imshow('Bitwise XOR', bitwise_xor) 

# Bitwise NOT inverts black to white and white to black 
bitwise_not = cv2.bitwise_not(rectangle) 
cv2.imshow('Bitwise NOT', bitwise_not) 

cv2.waitKey(0) 