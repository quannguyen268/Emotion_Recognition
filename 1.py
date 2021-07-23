from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
import numpy as np
import visualkeras
from PIL import ImageFont


font = ImageFont.truetype("/home/quan/.config/google-chrome/Default/Extensions/fhbjgbiflinjbdggehcddcbncdddomop/5.5.5_0/assets/fonts/OpenSans/OpenSans-Regular.ttf", 32)
# parameters for loading data and images
emotion_model_path = '/home/quan/PycharmProjects/Emotion_recog/__results___files/fer2013_7.model'

# hyper-parameters for bounding boxes shape
# loading models
emotion_classifier = load_model(emotion_model_path, compile=False) # model nhận diện cảm xúc


visualkeras.layered_view(emotion_classifier,to_file='output.png', legend=True, font=font).show()  # font is optional!
