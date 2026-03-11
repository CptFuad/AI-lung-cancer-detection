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



#gradcam function

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap,0) / np.max(heatmap)

    return heatmap.numpy()

# GradCAM
heatmap = make_gradcam_heatmap(img, model, "top_conv")

heatmap = cv2.resize(heatmap, (224,224))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + np.array(image.resize((224,224)))

st.image(superimposed_img.astype("uint8"), caption="Heatmap Visualization")