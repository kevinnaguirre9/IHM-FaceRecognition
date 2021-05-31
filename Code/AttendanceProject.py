import cv2
import face_recognition
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

# 1. Import images 
## We will create a list that will get the images from our folder automatically 
## and then it will generate the enconding for them automatically
## and them it will try to find it in our webcam

path = "../imagesAttendance"
images = []
classNames = []
userIDs = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
     curImg = cv2.imread(filename=f'{path}/{cl}')      ## load images, same as face_recognition.load_image_file
     images.append(curImg)                             ## add images to the list
     userIDs.append(int(re.findall(r'\d+', cl)[0]))    ## get the ids of the users
     name = os.path.splitext(cl)[0]                    ## get the name of the image without the .jpg
     classNames.append(re.sub(r'[0-9]', '', name))     ## remove the id of the image to keep just the name

## 2. Create a function that compute the encondings for us

def findEncodings(images_to_encode):
     encodeList = []
     for image in images_to_encode:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          encodeImg = face_recognition.face_encodings(image)[0]
          encodeList.append(encodeImg)
     return(encodeList)


def markAttendance(user_id, name):
     with open("../TempData/Attendance.csv", "a") as f:
          now = datetime.now()
          current_date = now.strftime("%d/%m/%Y") # obtenemos fecha
          current_time = now.strftime("%H:%M:%S") ## obtenemos hora

def markAttendance(path, user_id, name, temp):
     df = pd.read_csv(path)
     now = datetime.now()
     current_date = now.strftime("%d/%m/%Y")
     current_time = now.strftime("%H:%M:%S") 
     filtered_df = df[(df['userID'] == user_id) & (df['date'] == current_date)]
     if filtered_df.shape[0] == 0:
          with open(path, "a") as f:
               f.writelines(f'\n{user_id},{name},{current_date},{current_time},{temp}')

## Test function
encodeListKnownFaces = findEncodings(images)
#print('Encoding complete')
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
          #print(faceDis)
          matchIndex = np.argmin(faceDis) ## get the index of the min value 

          name = 'Unknown'
          temp = float(np.random.randint(36, 40))

          ## Display a bounding box around and write the name of the person
          if matches[matchIndex]:  ## si hay un TRUE en la posición de la mínima distancia
               name = classNames[matchIndex].upper() ## get the name of the match
               userID = userIDs[matchIndex]
               markAttendance("../TempData/Attendance.csv", userID, name, temp)
               #print(name)

          nameColor = (255, 0, 0)
          tempColor = (255, 0, 0)

          y1, x2, y2, x1 = faceLoc
          y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
          cv2.rectangle(img, (x1, y1), (x2, y2), (0,255, 0), 2)
          cv2.rectangle(img, (x1, y2+70), (x2, y2), (0,255, 0), cv2.FILLED)

          if name == "Unknown":
               nameColor = (0, 0, 255)
          if temp > 37.0:
               tempColor = (0, 0, 255)

          cv2.putText(img, name, (x1+6, y2+30), cv2.FONT_HERSHEY_COMPLEX, 1, nameColor, thickness=2)
          cv2.putText(img, str(temp), (x1+6, y2+60), cv2.FONT_HERSHEY_COMPLEX, 1, tempColor, thickness=2)
     cv2.imshow('Webcam', img)
     cv2.waitKey(1)