import os
import base64
import requests
from io import StringIO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

logo_url = 'detection_logo.png'
st.image(logo_url, width=150)

st.title('DetectionApp')
st.markdown("#### This app is a tool to classify x-ray images. :lungs:")
st.markdown(":warning: Attention: this app should only be used for **teaching purposes**\
     it should not be used for real diagnoses.")
# Criação do uploader de arquivos
uploaded_file = st.file_uploader(label="Choose an image...", type=['.jpg', '.png'])

if uploaded_file is None:
    st.image('xray_picture.png', width=100, caption='Your image here')
else:
    st.image(uploaded_file, width=200)

    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

    payload = {
         "image": encoded_image
         }

    api_url = os.getenv('URL')

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        if response.json()['status'] == 'fail':
            st.error("Failed to send the image. Try again later.")
        else:
            st.success('Image sent successfully.')
            st.markdown(f"#### Class: {response.json()['class']}")

            proba = pd.DataFrame(response.json()['probability'])

            df = pd.DataFrame(proba.T.values,
                              columns=['Covid-19', 'Normal', 'Opacity', 'Pneumonia'],
                              index = ['Probability'])

            st.write(df)

            fig = plt.bar(list(df.keys()), list(df.values[0]))
            st.pyplot(fig)
    else:
        st.error("Failed to send the image. Try again later.")
            
