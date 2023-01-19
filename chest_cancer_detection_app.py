import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

process_dir = os.getcwd()

def load_image(image_file):
    img = Image.open(image_file)
    return img
    
def chest_cancer_predict(image):
    
    if "png" in str(image):
        im = Image.open(image)
        rgb_im = im.convert('RGB')
        img2 = os.path.join(process_dir, "test.jpg")
        rgb_im.save(img2)

    else:
        img1 = Image.open(image)
        img2 = os.path.join(process_dir, "test.jpg")
        img1.save(img2)
                    
    img = cv2.imread(img2)    
    chest_cancer_model=load_model("chest_cancer_detector3.h5")
    test_image = cv2.resize(img,(224,224))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255    
    result = chest_cancer_model.predict(test_image)
#    print(result)
    
    if result[0][0] > 0.8:
        prediction = "adenocarcinoma cancer"
        confidence_score = result[0][0]*100
    elif result[0][1] > 0.8:
        prediction = "large cell carcinoma cancer"
        confidence_score = result[0][1]*100
    elif result[0][2] > 0.8:
        prediction = "normal chest"
        confidence_score = result[0][2]*100  
    else:
        prediction = "squamous cell carcinoma cancer"
        confidence_score = result[0][3]*100
        
    return prediction, confidence_score


def main():
    st.title("***Chest Cancer Prediction")
    html_temp = """
    <div style="background-color:red;padding:10px">
    <h2 style="color:white;text-align:center;">AI App for Chest Cancer Prediction</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
     
    filename = st.file_uploader("Upload Chest CT Scan Image", type=["jpg","png"])
#    print(filename)
    
    if ((filename is not None) and ('jpg' in str(filename))):
        st.image(load_image(filename))
        prediction, confidence_score = chest_cancer_predict(filename)
        confidence_score = str(confidence_score)

    elif ((filename is not None) and ('png' in str(filename))):
        st.image(load_image(filename))
        prediction, confidence_score = chest_cancer_predict(filename)
        confidence_score = str(confidence_score)

    else:
        pass
    
    result=""
    if st.button("Predict"):
        st.success('Prediction: {},   Confidence Score(%): {}'.format(prediction, confidence_score))
    


if __name__=='__main__':
    main()

