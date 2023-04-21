import face_recognition as fr
import cv2
import numpy as np
import os
 
face_path = "C:\\Users\\cohen\\OneDrive\\Desktop\\Ohene\\herbert"  #change this to your project folder


def get_face_encodings():
    face_names = os.listdir(f"{face_path}\\pics")
    # print(face_names)
    face_encoding = []
    
    for i, name in enumerate(face_names):
        face = fr.load_image_file(os.path.join(f"{face_path}\\pics", name))
        #print(fr.face_encodings(face))
        encodes = fr.face_encodings(face)

        if(len(encodes) > 0):
            face_encoding.append(encodes[0])
            face_names[i] = name.split(".")[0]

    return face_encoding, face_names

face_encodings, face_names = get_face_encodings()

video = cv2.VideoCapture(0)

scl= 1

while True:
    success, image = video.read()

    # if image is not None:
    resized_image = cv2.resize(image, (int(image.shape[1]/scl), int(image.shape[0]/scl)))
    # resized_image = cv2.resize(image, (28,28), fx=0.25, fy=0.25)

    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(rgb_image)
    
    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    imgdraw = image

    for face_encoding, face_locations in zip(unknown_encodings, face_locations):
        result = fr.compare_faces(face_encodings, face_encoding, 0.5 )

        

        if True in result:
            name  = face_names[result.index(True)]

            top, right, bottom, left = face_locations

            imgdraw = cv2.rectangle(resized_image, (left, top), (right, bottom), (0,255,0), 2)
            # cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 255, 0), 2)
            # cv2.rectangle(image, (0, 0), (200, 200), (0, 0, 0), -1)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(resized_image, name, (left, bottom+25), font, 1.0, (255, 255, 255), 1)
        else:
            name  = "Unknown Person"

            top, right, bottom, left = face_locations

            imgdraw = cv2.rectangle(resized_image, (left, top), (right, bottom), (0,255,0), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(resized_image, name, (left, bottom+25), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("frame", imgdraw)
    cv2.waitKey(1)
   
    




