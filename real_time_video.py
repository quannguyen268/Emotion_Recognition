from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from tensorflow.keras.applications import inception_v3
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# nhận dạng cảm xúc chia làm 2 bài toán : 1, nhận diện khuôn mặt   2, nhận diện cảm xúc

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml' # link dẫn model nhận diện khuôn mặt của open-cv, hôm sau sẽ thay bằng api của google
emotion_model_path = '/home/quan/PycharmProjects/Emotion_recog/models/fer2013_7.model' # link dẫn model nhận diện cảm xúc

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path) # model detect face
emotion_classifier = load_model(emotion_model_path, compile=False) # model nhận diện cảm xúc
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0) # lấy ảnh từ camera
while True:
    frame = camera.read()[1] # ảnh của mình
    #reading the frame
    frame = imutils.resize(frame,width=300) # chinhr lại kích thước ảnh về 300*300 vì model cascade nhận diện khuôn mặt của opencv nhận dữ liệu kích thước 300*300
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) # detect khuôn mặt trong frame
    
    canvas = np.zeros((250, 300, 3), dtype="uint8") # tạo một hình chữ nhật để chứa xác suất cảm xúc
    frameClone = frame.copy()

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces # tọa độ của khuôn mặt: (x,y) - góc trên bên trái của hình chữ nhật, (w, h) - chiều dài và rộng cần v
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi) # chuyển ma trận vuông 48x48 thành chuỗi có chiều dài : 2304 giá trị
        roi = np.expand_dims(roi, axis=0) # tăng thêm 1 chiều cho dữ liệu (1, 2304)
        
        
        preds = emotion_classifier.predict(roi)[0] # dự đoán khuôn mặt này đang có cảm xúc gì

        emotion_probability = np.max(preds) # xác suất cảm xúc lớn nhất mà khuôn mặt đó đang mang
        label = EMOTIONS[preds.argmax()] # lấy ra label được dự đoán
    else: continue

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100) # text lấy từ EMOTIONS bên trên để hiển thị

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]

                
                w = int(prob * 300) # xác suất kết quả rơi vào cảm xúc nào
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1) # vẽ hình chữ nhật màu đỏ
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2) # đặt chữ lên trên hình chữ nhật để hiển thị cảm xúc
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2) # vẽ hình chữ nhật bao khuôn mặt lại

    frameClone = cv2.resize(frameClone, (560, 500))
    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()




