import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# User inputs --
image_size = 256 # Size that the model reduces the images to during pre-processing
num_epochs = 10
reset = True # When set to false, ensure the same number of classes are used and in the same order as previously
# --------------

# Creating the model! (feel free to tweak)
if reset:
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='sigmoid')
        # Final output layer is excluded here, added in once the classes are known
    ])

else:
    model = keras.models.load_model('cnn_model.keras')

# Lists to store the photos and labels
pics, labels, label_names = [], [], [input('Choose a label name for the first class: ')]

# Initialize the webcam
cap = cv2.VideoCapture(1)
[cap.read() for i in range(50)] # Allow the camera to adjust to lighting

# Photo taking loop, type "stop" to exit, enter key to take a photo, anything else to define the next class --
while label_names[-1].lower() != 'stop':

    print(f'\nTaking photos of {label_names[-1]}')

    label = ''
    while label == '':
        label = input(f'Press enter to take a photo, enter the next class name to move on or \"stop\" to exit ({labels.count(len(label_names) - 1)} photos): ')
        
        if label != '':
            break

        ret, frame = cap.read()
        pics.append(frame)
        labels.append(len(label_names) - 1)

    label_names.append(label)
# ------------------------------------------------------------------------------------------------------------

# Release the webcam
cap.release()
cv2.destroyAllWindows()

# Adding the output layer of the model now that the class labels are known
if reset:
    model.add(keras.layers.Dense(len(label_names)-1, activation='softmax'))

def generate_data(image, N, size=64):
    """
    Function to generate data points from photos

    args:
        image (array like): the input photo
        N (int): number of extra data points to generate from the photo
        size (int): the size to reduce the image to, assumed square
    """
    image = cv2.resize(image, (size, size))
    output = [image]

    (h, w) = image.shape[:2]
    centre = (w // 2, h // 2)

    # Generating new data by rotating, scaling, and adding noise --
    for n in range(N):
        angle = np.random.normal(0, 5)
        scale = np.clip(np.random.normal(1, 0.05), 0.0, 2.0)
        M = cv2.getRotationMatrix2D(centre, angle, scale)
        T = np.float32([ [1, 0, np.clip(np.random.normal(0, 5), -int(size/2), int(size/2))], [0, 1, np.clip(np.random.normal(0, 5), -int(size/2), int(size/2))] ])
        noise = np.random.uniform(-10, 10, size=(size, size, 3)).astype(np.uint8)
        
        # Perform the rotation
        alt_image = np.clip(cv2.warpAffine(cv2.warpAffine(image + noise, M, (w, h)), T, (w, h)), 0, 255)
        output.append(alt_image)
    # -------------------------------------------------------------
    return output

# Generating the training data
X = []
y = []
for pic, label in zip(pics, labels):
    X.extend(generate_data(pic, 10, image_size))
    y.extend([label for l in range(11)])

X = np.array(X) / 255.0
y = np.array(y)

# Training the model --

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
print(f'\n{len(X)} data points for training')
model.summary()
model.fit(X, y, epochs=num_epochs)
model.save('cnn_model.keras') # Saving the trained model

# Writing the class labels to a txt file
with open('class_names.txt', 'w') as file:
    for l in label_names:
        if l != 'stop':
            file.write("\'" + l + "\', ")
        else:
            file.write("\'stop\'")

# --------------------
