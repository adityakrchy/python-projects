import cv2

face_capture = cv2.CascadeClassifier("env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

video_cap = cv2.VideoCapture(0)
while True:
    ret, video = video_cap.read()
    col = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    faces = face_capture.detectMultiScale(col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x, y, h, w) in faces:
        cv2.rectangle(video, (x, y), (x+h, y+w), (0, 255, 0), 2)
    cv2.imshow('Video', video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()