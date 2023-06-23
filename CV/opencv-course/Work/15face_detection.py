import cv2 

# Face detection simply recognizes if there is a face in an image, while 
# Face recognition recognizes whose face it is 

# OpenCV comes with a lot of pretrained classifiers 
# 2 Types: Haar Cascades & Local Binary Patterns

img = cv2.imread('Resources/Photos/group_1.jpg')
cv2.imshow("Group", img) 

# Convert to grayscale since we detect faces using only edges (no need for color) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Person", gray)

haar_cascade = cv2.CascadeClassifier('./Work/haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, 
                                           scaleFactor=1.1, 
                                           minNeighbors=1)

# faces_rect contains all the faces 
print(f'Number of Faces Found = {len(faces_rect)}')

# We loop over the faces and draw a rectangle about them 
for (x, y, w, h) in faces_rect: 
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    
cv2.imshow('Detected Faces', img)

# Haar Cascades are very sensitive to noise, so we should tweak the 
# scaleFactor and minNeighbors (high minNeighbors = more sensitive to faces
# and noise)

cv2.waitKey(0) 