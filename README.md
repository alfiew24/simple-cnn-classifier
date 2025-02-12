# Simple CNN Classifier

This short project was done to explore a simple application of convolutional neural networks on image classification via the ```Tensorflow``` and ```OpenCV``` libraries.
The project involves taking several photos of a given person/object, for each person/object that you wish for the model to distinguish between. More data points are then generated for each image through random transformations. The data is then fed to a simple CNN classifier for training. Following training, the application will open up a continuous camera feed and classify the camera feed at every frame.

Step-by-step running procedure:
1. Open the ```training.py``` file
2. In the file, check that you are happy with the model architecutre in lines 17-26, ajust where necassary
3. Also check that the user inputs defined in lines 9-12 are also preferrable, adjust where necassary
4. Run the file:
\t a. In the command prompt, type the name of the first class
\t b. Press the [enter key] to take photos for the class
\t c. When finished, type the name of the next class to classify
\t d. Repeat steps 4.b-4.c until all classes have been photographed, then type "stop"
5. The model will then be trained and saved
6. Open the ```testing.py``` file
7. Run the file!
