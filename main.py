import cv2
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('models/mouth.xml')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        mouths = mouth_cascade.detectMultiScale(gray,1.5,5)
        if len(mouths) == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img,'MASK DETECTED', (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            print("MASK DETECTED")
        else:
            for (mx, my, mw, mh) in mouths:
                if (y < my < y + h):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, 'MASK NOT DETECTED',  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("MASK NOT DETECTED")
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()