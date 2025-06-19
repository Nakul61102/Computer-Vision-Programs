import cv2
%run util.ipynb

yellow = [0, 255, 255]

# IP webcam port or any other app for different device
camera_url = 'http://0:0:0:0/video'

cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    cap = cv2.VideoCapture(0)

 
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lowerLimit, upperLimit = get_limits(color=yellow)
    
    mask = cv2.inRange(img_hsv, lowerLimit, upperLimit)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            # cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)

            x1, y1, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    


cap.release()
cv2.destroyAllWindows()
