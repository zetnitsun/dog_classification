import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import requests
from PIL import Image
import altair as alt
import datetime 
import numpy as np 
import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
import shutil
from shutil import copyfile
import pickle
import cv2

st.set_page_config(
     page_title="Dog Breed Classifier",
     page_icon="🐶",
     layout="wide",
     initial_sidebar_state="collapsed",
 )

st.title("Dog Breed Classifier 🐶")
st.markdown("_Made by Jessica Zerlina Sarwono_")

st.info('Upload a dog picture below to know their breed 😊')
image_up = st.file_uploader("", type=['jpg','png','jpeg'])

json_file = open('model_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("my_model_weights.h5")

if image_up is not None:
    st.image(Image.open(image_up), width = 600)
    uploaded = Image.open(image_up)
    x = img_to_array(uploaded)
    x = cv2.resize(x, (224,224))
    x /= 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = loaded_model.predict(images, batch_size=10)
    breeds = ['Beagle', 'Bulldog', 'Dalmatian','German Sheperd','Husky','Labrador Retriever','Poodle','Rottweiler']
    result = pd.DataFrame(list(zip(breeds,classes[0])))
    result.columns = ["Breed", "Probability"]
    breed = result[result["Probability"] == result["Probability"].max()]
    breed = breed["Breed"].to_string(index = False)
    st.write("There is a high chance that your dog is a " + breed)


    if st.button('Really? Show me the full result!'):
        sorted = result.sort_values(by = "Probability", ascending = False)
        for i in range(len(sorted)):
            if round(sorted.iloc[i,1],2) > 0:
                st.write("Your dog could be a " + str(sorted.iloc[i,0]) + " with a probability of " + str(round(sorted.iloc[i,1],2)))
else:
    st.write("You have to upload a picture to get started!")
        
information = st.sidebar.markdown("_Application Version 1.0_")
st.sidebar.markdown("_This app is trained on [Kaggle's Dog Breeds dataset](https://www.kaggle.com/datasets/mohamedchahed/dog-breeds) using transfer learning. For this version, the dog breeds are limited to beagle, bulldog, dalmatian, german shepherd, husky, labrador retriever, poodle, and rottweiler._")
st.sidebar.markdown("_More application and data development coming soon!_")
