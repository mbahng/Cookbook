import cv2 as cv 

def rescaleFrame(frame, scale=0.75): 
    width = int(frame.shape[1] * scale) 
    height = int(frame.shape[0] * scale) 
    
    dimensions = (width, height) 
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) 

capture = cv.VideoCapture('./Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame, 0.5) 
    
    # if cv.waitKey(20) & 0xFF==ord('d'):
    # This is the preferred way - if `isTrue` is false (the frame could 
    # not be read, or we're at the end of the video), we immediately
    # break from the loop. 
    if isTrue:    
        cv.imshow('Video', frame)
        
        cv.imshow('Rescaled Video', frame_resized)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break            
    else:
        break

capture.release()
cv.destroyAllWindows()