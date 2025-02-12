import tensorflow as tf
import cv2
import numpy as np

# Loading the model and class labels --
model = tf.keras.models.load_model('cnn_model.keras')
image_size = model.input_shape[1]

with open('class_names.txt', 'r') as file:
    labels = eval('['+file.read()+']')
# -------------------------------------

# Initialize the webcam and create the window
cap = cv2.VideoCapture(1)
cv2.namedWindow('Recognition Model')

def recognise():
    """
    This function takes a snapshot from the camera, processes the image and puts it through the model.
    The outputs from the model are displayed in the window.
    """
    # Taking snapshot, processing, and predicting --
    ret, frame = cap.read()
    x = cv2.resize(frame, (image_size, image_size)) / 255.0
    result = model.predict(np.array([x]), verbose=0)[0]
    choice = result.argmax()
    # ----------------------------------------------

    text_lines = []
    colours = []
    widths = []

    # Generating the text for every class
    for i in range(len(result)):
        text = f'{labels[i]} {result[i]*100:.2f}%'
        text_lines.append(text)
        if i == choice:
            colours.append((200, 255, 200))
        else:
            colours.append((100, 100, 100))

        widths.append(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0])
        
    # Adding a rectangle to sit behind the text
    shapes = np.zeros_like(frame, np.uint8)
    cv2.rectangle(shapes, (0, 30 + 20*i), (max(widths)+10, 0), (20, 20, 20), -1)
    mask = shapes.astype(bool)
    frame[mask] = cv2.addWeighted(frame, 0.5, shapes, 0.5, 0)[mask]

    # Displaying the text
    for i, text in enumerate(text_lines):
        cv2.putText(frame, text, (5, 20 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours[i], 1, cv2.LINE_AA)
    
    # Displaying the entire image, text and all!
    cv2.imshow('Recognition Model', frame)

camera = 1
while cv2.getWindowProperty('Recognition Model', 0) >= 0:
    recognise()
    key = cv2.waitKey(20)
    if key == 27: # esc key - to exit
        break
    elif key == 115: # s key - used to swap to an alternative camera
        cap.release()
        camera = (camera + 1) % 3
        cap = cv2.VideoCapture(camera)

# Release the webcam
cap.release()
cv2.destroyAllWindows()
