import cv2 

img = cv2.imread('Resources/Photos/cats.jpg')
cv2.imshow('Cats', img) 

# Many types of blurs with different kernels 
# Averaging blur 
average = cv2.blur(img, (3, 3))
cv2.imshow("Average Blur", average)

# Gaussian blur 
gauss = cv2.GaussianBlur(img, (3, 3), 0) 
cv2.imshow("Gaussian Blur", gauss)

# Median blur (finds median of pixels), more effective in reducing noise 
median = cv2.medianBlur(img, 3) 
cv2.imshow("Median Blur", median) 

# Bilateral blurring (most effective) applies blurring but retains edges 
bilateral = cv2.bilateralFilter(img, 10, 15, 15) 
cv2.imshow("Bilaterial Blur", bilateral) 

cv2.waitKey(0)
