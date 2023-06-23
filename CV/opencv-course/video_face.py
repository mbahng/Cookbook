import cv2

vid = cv2.VideoCapture(0) 

while True: 
    success, frame = vid.read() 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    haar_cascade = cv2.CascadeClassifier('./Work/haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, 
                                           scaleFactor=1.1, 
                                           minNeighbors=7)
    
    for (x, y, w, h) in faces_rect: 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        
    dst = cv2.cornerHarris(gray,2,3,0.04)
    frame[dst>0.01*dst.max()]=[0,0,255]
    
    cv2.imshow('Detected Faces', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
    
vid.release() 
cv2.destroyAllWindows() 