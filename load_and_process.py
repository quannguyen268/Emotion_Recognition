import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer

dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)

def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels']
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [pixel for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = data['emotion'].copy()
        emotions = LabelBinarizer().fit_transform(emotions)
        return faces, emotions

def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

x = []
