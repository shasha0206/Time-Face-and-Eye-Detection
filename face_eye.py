import cv2

face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cam = cv2.VideoCapture(0)

while True:
    ret,image = cam.read()
    gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray_face, 1.05, 5)#1.05 scale its recomendation

    for (x,y,w,h) in  faces:                 
        cv2.rectangle(image,(x,y),(x + w , y + h),(0,255,0),5)

        gray_eye = gray_face[y:y + h, x:x+w]
        gray_c = image[y:y + h , x:x+w]

        eyes = eye.detectMultiScale(gray_eye,1.05,5)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(gray_c,(ex,ey),(ex + ew , ey + ew),(255,0,0),5)
            
    cv2.imshow('Face recognition',image)

    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()
