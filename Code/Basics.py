import cv2
import numpy as np
import face_recognition

## Load image and convert it into rgb
imgElon = face_recognition.load_image_file('ImagesBasic/Elon-Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

## Do the same with test image
imgTest = face_recognition.load_image_file('ImagesBasic/elon-test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


## Find the exact locations of the faces in our images and their enconding
faceLoc = face_recognition.face_locations(imgElon)[0]
encondeElon = face_recognition.face_encodings(imgElon)[0]
#print(faceLoc) ## prints out 4 values: top, right, bottom and left
cv2.rectangle(img=imgElon, pt1=(faceLoc[3], faceLoc[0]), pt2=(faceLoc[1], faceLoc[2]), color=(255, 0, 255), thickness=2)

## Repeat previuos step for test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encondeTest = face_recognition.face_encodings(imgTest)[0]
#print(faceLocTest) ## prints out 4 values: top, right, bottom and left
cv2.rectangle(img=imgTest, pt1=(faceLocTest[3], faceLocTest[0]), pt2=(faceLocTest[1], faceLocTest[2]), color=(255, 0, 255), thickness=2)


#Comparing the faces and finding the distance between them
results = face_recognition.compare_faces(known_face_encodings=[encondeElon], face_encoding_to_check=encondeTest)
#But, how similars are they. The lower de distance, the better the match is
faceDis = face_recognition.face_distance(face_encodings=[encondeElon], face_to_compare=encondeTest)
print(results, faceDis)
cv2.putText(img = imgTest, text=f'{results} {round(faceDis[0], 2)}', org=(50,50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255), thickness=2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)