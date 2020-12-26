import numpy as np
import cv2
import FaceRecognition as fr

test_img=cv2.imread('F:/master thesis/thesis/TestImages/sonali.jpg')
faces_detected, gray_img = fr.faceDetection(test_img) #identifying number of faces in the image
print("faces_detected:",faces_detected)

"""
#adding a rectangle to identy the faces
for(x, y, w, h) in faces_detected:
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness = 10)
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow("face detected:",resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
"""
#training the classifier

faces,faceID=fr.labels_for_training_data('F:/master thesis/thesis/TrainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('F:/master thesis/thesis/trainingData.yml')

name={0:"Random",1:"Sonali",2:"Rajesh"}

#feeding the video from camera
#cap = cv2.VideoCapture(0)
for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y : y + w, x : x + h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if (confidence>37):
        continue
    fr.put_text(test_img, predicted_name, x, y)    

    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow("face detected:",resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
cv2.destroyAllWindows
