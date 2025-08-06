import cv2
import numpy as np

# haar cascade 
haar_frontal_face_filters = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
frontal_face_detector = cv2.CascadeClassifier(haar_frontal_face_filters)

haar_profile_face_filters = cv2.data.haarcascades + 'haarcascade_profileface.xml'
profile_face_detector = cv2.CascadeClassifier(haar_profile_face_filters)

# LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("lbph_model.yml")
label_map = np.load("labels.npy", allow_pickle=True).item()
id_map = {v: k for k, v in label_map.items()}

cam = cv2.VideoCapture(0)

while True:

    ret, frame = cam.read()
    if not ret:
        print("Error on camera acces.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontal_faces = frontal_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    profile_faces = profile_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    # print("Valores do rosto: ", faces)

    for (x, y, w, h) in frontal_faces:
        roi = gray[y:y+h, x:x+w]
        id_predito, confianca = recognizer.predict(roi)

        if confianca < 100:
            nome = id_map[id_predito]
        else:
            nome = "Desconhecido"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        cv2.putText(frame, nome, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # for (x, y, w, h) in profile_faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 

    cv2.imshow("Face detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()