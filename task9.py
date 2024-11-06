import cv2, time, math
from picamera2 import Picamera2, Preview
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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

model = model = models.mobilenet_v3_small(weights=None, num_classes=2)
model.load_state_dict(torch.load('task_8_best_model.pth'))
model.eval()

while True:
    frame = picam.capture_array()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    text = ""
    
    if len(faces) > 0:
        x,y,w,h = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)

    cv2.putText(frame, text, (0,0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Preview", frame)

    if (x := cv2.waitKey(1)):
        if x & 0xFF == ord('p'):
            cropped_image = frame[y:y-h, x:x-w]
            scaled_image = cv2.resize(cropped_image, (64, 64))
            scaled_image = (scaled_image / 255.0)
            
            image_tensor = torch.tensor(scaled_image)
            image_tensor = image_tensor.double()
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
            with torch.no_grad():
                output = model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                if predicted_class == 0:
                    text = "Shakalan"
                else:
                    text = "Sean"  

        if x & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
picam.stop()