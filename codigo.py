#from util import classify, set_background
# import cv2
import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

#set_background('app/bgs/eye2.png')
# print(os.getcwd())
# set title
st.title('Diabetic Retinopathy classification')
# set header
st.header('Please upload a Retinal Scan Image')
# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
set_background('app/bgs/eye2.png')
# print(os.getcwd())
# set title
st.title('Diabetic Retinopathy classification')
# set header
st.header('Please upload a Retinal Scan Image')
# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


# URL del modelo en Google Drive
model_url = 'https://drive.google.com/file/d/1U96luzv8S4RLlUI6np_ZR7JgmM8QduA3/view?usp=sharing'
