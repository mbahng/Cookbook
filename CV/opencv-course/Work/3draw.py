import cv2 
import numpy as np 

# create blank image to work with (500 times 500 pixels, with RGB) 
blank = np.zeros((500, 500, 3), dtype='uint8')
# cv2.imshow('Blank', blank)

# Color a red rectangle in image 
# blank[200:300, 300:400] = 0, 0, 255
# cv2.imshow('Red square', blank)

# Draw a rectangle and a filled rectangle 
cv2.rectangle(blank, (260, 270), (400, 490), (0, 255, 0), thickness=2)
cv2.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=cv2.FILLED)
cv2.rectangle(blank, (0, 0), (blank.shape[1]//3, blank.shape[0]//2), (0, 255, 0), thickness=cv2.FILLED)
cv2.imshow("rectangle", blank) 

# Draw a circle: center, radius, color thickness 
cv2.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=3) 
cv2.imshow("circle", blank)

# Draw a line: start point, end point, color, thickness
cv2.line(blank, (0, 0), (45, 123), (242, 7, 43), thickness=3)
cv2.imshow("line", blank)

# Write text: text, location, font, font scale, color, thickness 
cv2.putText(blank, "Hello", (225, 225), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (98, 255, 340), 2) 
cv2.imshow("font", blank)

cv2.waitKey(0)