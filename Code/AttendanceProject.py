import cv2
from face_recognition.api import face_locations
import face_recognition_models
import numpy as np
import face_recognition
import os

# 1. Import images 
## We will create a list that will get the images from our folder automatically 
## and then it will generate the enconding for them automatically
## and them it will try to find it in our webcam

path = "imagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
     curImg = cv2.imread(filename=f'{path}/{cl}')      ## load images, same as face_recognition.load_image_file
     images.append(curImg)                             ## add images to the list
     classNames.append(os.path.splitext(cl)[0])        ## get the names of the images without the .jpg

print(classNames)

## 2. Create a function that compute the encondings for us

def findEncodings(images_to_encode):
     encodeList = []
     for image in images_to_encode:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          encodeImg = face_recognition.face_encodings(image)[0]
          encodeList.append(encodeImg)
     return(encodeList)

## Test function
encodeListKnownFaces = findEncodings(images)
print('Encoding complete')
##print(len(encodeListKnownFaces))


## 3. Find the matches between our encodings and our webcame 

cap = cv2.VideoCapture(0)

## get each frame one by one
while True:
     success, img = cap.read()
     # reduce size of our image to improve speeding the process
     imgSmall = cv2.resize(src = img, dsize=(0,0), dst=None, fx=0.25, fy=0.25)       ## 1/4 of the size
     imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

     # find all the face locations 'cause it could be there multiple faces
     facesCurFrame = face_recognition.face_locations(img=imgSmall)  
     ## find encodings of our webcam
     encodesCurFrame = face_recognition.face_encodings(face_image=imgSmall, known_face_locations=facesCurFrame)      
     

     ## 3. Iterate through all te faces we found in the current frame 
     ## and compare with all the encodings we found before

     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
          matches = face_recognition.compare_faces(known_face_encodings=encodeListKnownFaces, face_encoding_to_check=encodeFace)
          ## find the distance, it will return 5 values becasuse we have 5 known faces in our path
          ## The lower ditance will be the best match
          faceDis = face_recognition.face_distance(face_encodings=encodeListKnownFaces, face_to_compare=encodeFace)
          print(faceDis)
          matchIndex = np.argmin(faceDis) ## get the index of the min value 

          ## Display a bounding box around and write the name of the person
          if matches[matchIndex]:  ## si hay un TRUE en la posición de la mínima distancia
               name = classNames[matchIndex].upper() ## get the name of the match
               print(name)


