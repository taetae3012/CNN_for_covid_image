
import h5py
from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

# from skimage.io import imshow, imread
#from keras.models import load_model

Result = None

st.image('header.gif')

decodeLabel = {
    0: "Covid",
    1: "Normal",
    2: "Virus"
}


def load_model():
    model = tf.keras.models.load_model('Covid.h5')
    return model


def Pred(image_data, model):

    #img = cv2.imread(image_data)
    img = Image.open(image_data)
    #img = imread(image_data)
    image = img.convert("RGB")
    image = image.resize((256, 256))
    img = np.array(image, dtype='float32')
    img = img/255
    img = img.reshape((1, 256, 256, 3))
    res = decodeLabel[np.argmax(model.predict(img))]
    Result = res
    
    if Result != None:
        if Result == 'Normal':
            st.header("Congratulations, You are Normal")
            col1, col2, col3 = st.columns([5, 5, 5])

            with col1:
                st.header("Things to keep in mind")
                st.write(
                    '• Maintain Social Distancing\n\n• Eat healthy food\n\n• Wear Mask.',)
            with col2:
                st.image('normal.gif')

            with col3:
                st.write("")

            st.balloons()

        elif Result == 'Covid':
            st.header("You are Corona Positive")
            col1, col2, col3 = st.columns([5, 5, 3])

            with col1:
                st.write()
                st.header("Things to keep in mind")
                st.write(
                    'You are infected with the covid 19 virus.\n\n• Get vaccinated as soon as possible.\n\n• Isolate yourself from others.\n\n• Take rest and drink a lot of fluids for hydration',)
            with col2:
                st.image('covid.png')

            with col3:
                st.write("")

        else:
            st.subheader("You have viral Pnemonia")
            col1, col2, col3 = st.columns([5, 5, 3])
            with col1:
                st.write()
                st.header("Things to keep in mind")
                st.write(
                    'You are infected with Pneumonia.\n\n• Drink plenty of fluids to help loosen secretions and bring up phlegm.\n\n• Do not take cough medicines without first talking to your doctor.',)
            with col2:
                st.image('pnemonia.png')

            with col3:
                st.write("")


model = load_model()

img = st.file_uploader("Select a picture", type=['jpg', 'png', 'jpeg'])
if img is not None:
    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        st.write()
    with col2:
        st.image(img, width=300)
    with col3:
        st.write()
    Pred(img, model)