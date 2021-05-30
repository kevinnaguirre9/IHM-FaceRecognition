import cv2
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

## CONTNUAR EN MINUTO 25:32
print(os.path.splitext("Elon Musk.jpg")[:])