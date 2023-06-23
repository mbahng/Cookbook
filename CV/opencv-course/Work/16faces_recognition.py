import os
import cv2 
import numpy as np 

people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]
DIR = r'/home/mbahng/Desktop/opencv-course/Resources/Faces/train'

haar_cascade = cv2.CascadeClassifier('./Work/haar_face.xml')
features = [] 
labels = []  

def create_train(): 
    for person in people: 
        path = os.path.join(DIR, person) 
        label = people.index(person) 
        
        for img in os.listdir(path): 
            img_path = os.path.join(path, img) 
            
            img_array = cv2.imread(img_path) 
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) 
            
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            for (x, y, w, h) in faces_rect: 
                faces_roi = gray[y:y+h, x:x+w] 
                features.append(faces_roi)                
                labels.append(label)
                
create_train() 

# print(f"Length of Features = {len(features)}")
# print(f"Length of Labels = {len(labels)}")

print("Training Done -------------------------------")

# quickly convert the features and labels to numpy arrays 
features = np.array(features, dtype='object') 
labels = np.array(labels) 

# Instantiate a face recognizer object 
face_recognizer = cv2.face.LBPHFaceRecognizer_create() 

# Train the recognizer on the features list and the labels list (you can save this in a yaml file)
face_recognizer.train(features, labels)

img = cv2.imread(r"/home/mbahng/Desktop/opencv-course/Resources/Faces/train/Ben Afflek/9.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# cv2.imshow("Person", gray) 

# Detect the face in the image 
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4) 

for (x, y, w, h) in faces_rect: 
    faces_roi = gray[y:y+h, x:x+h] 
    
    label, confidence = face_recognizer.predict(faces_roi) 
    print(f"label = {people[label]} with confidence {confidence}")
    
    cv2.putText(img, str(people[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2) 
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) 
    
cv2.imshow("Detected Face", img) 

cv2.waitKey(0) 
    