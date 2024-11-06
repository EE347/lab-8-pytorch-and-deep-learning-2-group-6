import cv2, time, math
from picamera2 import Picamera2, Preview
from datetime import datetime
import numpy as np

picam = Picamera2()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

config = picam.create_still_configuration(
    main={"size": (1280, 720),
          "format": 'XRGB8888'},
    lores={"size": (640, 480)},
    display="lores"
)

picam.configure(config)
picam.start()

while True:
    frame = picam.capture_array()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        x,y,w,h = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)

    cv2.imshow("Preview", frame)

    if (x := cv2.waitKey(1)):
        if x & 0xFF == ord('p'):
            cropped_image = frame[y:y-h, x:x-w]
            scaled_image = cv2.resize(cropped_image, (64, 64))
            cv2.imwrite(f"task1-{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", scaled_image)

        if x & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
picam.stop()