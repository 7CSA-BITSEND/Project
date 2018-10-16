# include header files
import cv2
import numpy as np 
from PIL import Image
import os

#load the face detection cascade
fn_haar = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(fn_haar)

#creating the face Recognizer object
Recognizer = cv2.face.LBPHFaceRecognizer_create()

#function to prepare training sets
def get_images_and_labels(path):
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if f in os.listdir(path) if not f.endswith('.sad')]
	images = []
	labels = []
	for image_path in image_paths:
		# Read the image and convert to grayscale
		image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
	return images, labels
		
#preparing the training sets
#The folder database is in the same folder as this python script is.
path = 'yalefaces'

#Now call get_images_and_labels function and get the face images and corresponding _labels
images , labels = get_images_and_labels(path)
cv2.destroyAllWindows()

#Now perform the training session for the available data
Recognizer.train(images, np.array(labels))