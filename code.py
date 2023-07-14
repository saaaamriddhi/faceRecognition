## Dataset

import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces == ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
    
cap = cv2.VideoCapture(0)
count = 0

# Collect 200 samples of your face from webcam input
while True:

    ret, photo = cap.read()
    if face_extractor(photo) is not None:
        count += 1
        face = cv2.resize(face_extractor(photo), (200, 200)) #getting the face and changing size of image
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #convert colour image to black-white (grayscale)

        # Save file in specified directory with unique name
        file_name_path = 'E:\ML_Projects\DataSet\\' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        pass

    if cv2.waitKey(1) == 27 or count == 200: #27 is the Esc Key
        break
        
cap.release()
cv2.destroyAllWindows() #upon completion close the window      
print("Collecting Samples")
print("Training Data Set Complete")

## Model Train

# model creation
# python -m pip install --user opencv-contrib-python

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
print(cv2.__version__)
# Get the training data we previously made
data_path = 'E:\ML_Projects\DataSet\\'
# a=listdir('d:/faces')
# print(a)
# """
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
# 
# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)
# model=cv2.face_LBPHFaceRecognizer.create()

model = cv2.face.LBPHFaceRecognizer_create() 
# Initialize facial recognizer
#model = cv2.face_LBPHFaceRecognizer.create()
# model=cv2.f
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

# Let's train our model 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained successfully")


## Recognition

import cv2
import numpy as np
from os import system
import os

# Load face detector
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):

    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

# Deployment Launched or Not
launched = False


while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        print(results)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
        
        # Display Confidence Score
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (204,72,63), 2)

        if confidence > 80:
           
            cv2.putText(image, "Welcome Sayantan!!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            system("kubectl cluster-info > k8s_status.txt")
            
            file = open("k8s_status.txt", "r")
            k8s_status = str(file.read())
            file.close()
            
            if "running" in k8s_status:
                if launched == False:
                    cv2.putText(image, "Launching Deployment", (1, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,128,0),2)
                    system("kubectl create deployment test1 --image=httpd")
                    launched = True
                else:
                    cv2.putText(image, "Launched Deployment", (1, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,128,0),2)
                    launched = True
            cv2.imshow('Face Recognition', image )
            
        else:
            cv2.putText(image, "i dont know", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)            
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()


if os.path.exists("k8s_status.txt"):
    os.remove("k8s_status.txt")
