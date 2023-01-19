import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageEnhance
import os


#----------------------------------------------------
#img = cv2.imread('adenocarcinoma1.png')
#img = cv2.imread('large-cell-carcinoma1.png')
#img = cv2.imread('normal-chest1.png')
img = cv2.imread('squamous-cell-carcinoma1.png')


chest_cancer_model=load_model("chest_cancer_detector3.h5")

test_image = cv2.resize(img,(224,224))
test_image = np.array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255    
result = chest_cancer_model.predict(test_image)
print(result)

if result[0][0] > 0.8:
    prediction = "adenocarcinoma"
    confidence_score = result[0][0]
elif result[0][1] > 0.8:
    prediction = "large-cell-carcinoma"
    confidence_score = result[0][1]
elif result[0][2] > 0.8:
    prediction = "normal"
    confidence_score = result[0][2]  
else:
    prediction = "squamous-cell-carcinoma"
    confidence_score = result[0][3]

print()
print("Chest Cancer prediction = {}".format(prediction))
print("Confidence score(%) = {}".format(confidence_score*100))



