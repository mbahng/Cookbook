import cv2
import matplotlib.pyplot as plt

# Visualize the distribution of pixel intensities of an image 
# Can do with both grayscale and RGB images

img = cv2.imread('/home/mbahng/Desktop/Cookbook/CV/opencv-course/Resources/Photos/cats.jpg')
cv2.imshow("Cats", img) 

# Color histogram 
colors = ('b', 'g', 'r') 
for i, col in enumerate(colors): 
    hist = cv2.calcHist([img], [i], None, [256], [0, 256]) 
    plt.plot(hist, color=col) 
    plt.xlim([0, 256]) 
    
plt.show() 
cv2.waitKey(0) 

assert False 


# Convert to grayscale q
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Gray', gray) 

# Grayscale Histogram for a list of image
# gray_hist = cv2.calcHist(images=[gray], 
#                          channels=[0], 
#                          mask=None, 
#                          histSize=[256], 
#                          ranges=[0, 256])

# plt.figure() 
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.plot(gray_hist) 
# plt.xlim([0, 256])
# plt.show() 


