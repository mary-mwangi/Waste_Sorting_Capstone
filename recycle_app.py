import streamlit as st 
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def preprocess_image(new_image):
  test_image = image.img_to_array(new_image) / 255
  test_image = tf.image.resize(test_image, size=[224, 224])
  test_image = np.expand_dims(test_image, axis=0)

  return test_image


st.title("Waste Classification")
st.subheader("Classifying Waste Materials for Recycling")


model = tf.keras.models.load_model('vgg.h5')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)

    uploaded_image = preprocess_image(uploaded_image)

    if st.button("Classify:"):
        pred = model.predict(uploaded_image)
        label = np.argmax(pred, axis=1)
        item = class_names[label[0]]

        if item == "trash":
            st.write("This item is " + item  + ". Your waste material is not recylable")
        else:
            st.write("This item is " + item  + ". Your waste material is recylable")

st. subheader("Quick Facts")
st.write("Glass is 100% recyclable! As it is easy to wash off contaminations, it is the most likely to be recycled multiple times. Simply ensure to not break into tiny pieces")
st.write("To go cups are not recyclable. Do not confuse them with paper or cardboard ")
st.write("Water is a contaminant. Remember to close your bin in case of stormy weather.")


st. subheader("Useful Links")

st.write("Recycling 101")
col1, mid, col2 = st.beta_columns([1,1,20])
with col1:
    st.image('row_2_col_2.jpg', width=60)
with col2:
    st.write('https://www.youtube.com/watch?v=kozJmBpLDF8')


st.write("How to Properly Reycle At Home")
col1, mid, col2 = st.beta_columns([1,1,20])
with col1:
    st.image('row_2_col_1.jpg', width=60)
with col2:
    st.write('https://www.youtube.com/watch?v=jsp7mgYv3aI')


