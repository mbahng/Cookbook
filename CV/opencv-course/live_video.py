import cv2

vid = cv2.VideoCapture(0, cv2.CAP_V4L2) 

while True: 
    success, frame = vid.read() 
    
    cv2.imshow('frame', frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
    
vid.release() 
cv2.destroyAllWindows() 