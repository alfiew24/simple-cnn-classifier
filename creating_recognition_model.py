import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
TF_ENABLE_ONEDNN_OPTS = 0

reset = False

pics, labels, label_names = [], [], [input('Choose a label name for the first class: ')]

# Initialize the webcam
cap = cv2.VideoCapture(1)
[cap.read() for i in range(50)]

while label_names[-1] != 'stop':

    print(f'\nTaking photos of {label_names[-1]}')

    label = ''
    while label == '':
        label = input('Press enter to take a photo, enter the next class name to move on or \"stop\" to exit: ')
        
        if label != '':
            break

        ret, frame = cap.read()
        pics.append(frame)
        labels.append(len(label_names) - 1)

    label_names.append(label)

# Release the webcam
cap.release()
cv2.destroyAllWindows()

def generate_data(image, N, size=64):
    image = cv2.resize(image, (size, size))
    output = [image]

    (h, w) = image.shape[:2]
    centre = (w // 2, h // 2)

    for n in range(N):
        angle = np.random.normal(0, 5)
        scale = np.clip(np.random.normal(1, 0.05), 0.0, 2.0)
        M = cv2.getRotationMatrix2D(centre, angle, scale)
        T = np.float32([ [1, 0, np.clip(np.random.normal(0, 5), -int(size/2), int(size/2))], [0, 1, np.clip(np.random.normal(0, 5), -int(size/2), int(size/2))] ])
        noise = np.random.uniform(-10, 10, size=(size, size, 3)).astype(np.uint8)
        
        # Perform the rotation
        alt_image = np.clip(cv2.warpAffine(cv2.warpAffine(image + noise, M, (w, h)), T, (w, h)), 0, 255)
        output.append(alt_image)
    return output

X = []
y = []
for pic, label in zip(pics, labels):
    X.extend(generate_data(pic, 10, 128))
    y.extend([label for l in range(11)])

X = np.array(X) / 255.0
y = np.array(y)

if reset:
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(len(label_names)-1, activation='softmax')
    ])

else:
    model = keras.models.load_model('recognition.keras')

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
print(len(X))
model.fit(X, y, epochs=10)
model.save('recognition.keras')

#k = 0
#for pic, l, l_pred in zip(X, y, model.predict(X).argmax(axis=1)):
#    if k % 10 == 0:
#        cv2.imshow(label_names[l] + ' | ' + label_names[l_pred], cv2.resize(pic, (256, 256)))
#        cv2.waitKey(0)
#    k += 1

with open('class_names.txt', 'w') as file:
    for l in label_names:
        if l != 'stop':
            file.write("\'" + l + "\', ")
        else:
            file.write("\'stop\'")
