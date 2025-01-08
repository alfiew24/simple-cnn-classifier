import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('recognition.keras')

with open('class_names.txt', 'r') as file:
    labels = eval('['+file.read()+']')

cv2.namedWindow('Recognition Model')

# Initialize the webcam
cap = cv2.VideoCapture(1)

def recognise():
    #[cap.read() for i in range(10)]

    ret, frame = cap.read()
    x = cv2.resize(frame, (128, 128)) / 255.0
    result = model.predict(np.array([x]), verbose=0)[0]

    choice = result.argmax()

    for i in range(len(result)):
        text = f'{labels[i]} {result[i]*100:.2f}%'
        colour = (100, 100, 100)
        if i == choice:
            colour = (200, 255, 200)

        #sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        #cv2.rectangle(frame, (0, 20 + 20*i), sz, (0, 0, 0), -1)
        cv2.putText(frame, text, (5, 20 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    
    cv2.imshow('Recognition Model', frame)

camera = 1
while True:
    recognise()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 115:
        cap.release()
        camera = (camera + 1) % 2
        cap = cv2.VideoCapture(camera)

# Release the webcam
cap.release()
cv2.destroyAllWindows()
