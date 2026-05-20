import cv2
import numpy as np
from picamera2 import Picamera2
import serial
import time

# Connect Arduino (CHANGE if needed)
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

# Load model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant",
           "sheep","sofa","train","tvmonitor"]

# Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (640,480)}
))
picam2.start()

print("Dog detection started...")

while True:
    frame = picam2.capture_array()
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    dog_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            idx = int(detections[0,0,i,1])

            if CLASSES[idx] == "dog":
                dog_detected = True

                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX,startY,endX,endY) = box.astype("int")

                label = f"Dog: {confidence*100:.2f}%"

                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
                cv2.putText(frame,label,(startX,startY-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # Send signal to Arduino
    if dog_detected:
        arduino.write(b'1')  # buzzer ON
    else:
        arduino.write(b'0')  # buzzer OFF

    cv2.imshow("Dog Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
arduino.close()
