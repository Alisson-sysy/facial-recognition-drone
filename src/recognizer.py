import cv2
import os
import numpy as np


haar_frontal_face_filters = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
frontal_face_detector = cv2.CascadeClassifier(haar_frontal_face_filters)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels =  []
label_map = {}
label_id = 0

base_directory = "../data"

for people in os.listdir(base_directory):
    people_directory = os.path.join(base_directory, people)
    if not os.path.isdir(people_directory):
        continue

    if people not in label_map:
        label_map[people] = label_id
        label_id += 1

    for img_name in os.listdir(people_directory):
        img_path = os.path.join(people_directory, img_name)
        img_gray_scale = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    faces_detection = frontal_face_detector.detectMultiScale(img_gray_scale, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detection:
        roi = img_gray_scale[y:y+h, x:x+w]
        faces.append(roi)
        labels.append(label_map[people])

recognizer.train(faces, np.array(labels))
recognizer.save("lbph_model.yml")
np.save("labels.npy", label_map)