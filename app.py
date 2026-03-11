import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

st.title('AI Lung Cancer detection ')

model=tf.keras.models.load_model('model/eff.h5')
class_names=['Normal','Benign', 'Malignant']

uploaded_file=st.file_uploader('Upload the CT Scan Image', type=['jpg', 'png'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded CT Scan Image')

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    st.write('Prediction class is: ', predicted_class)

    prob = prediction[0]

    for i in range(len(class_names)):
        st.write(class_names[i], ':', round(prob[i]*100,2), '%')
