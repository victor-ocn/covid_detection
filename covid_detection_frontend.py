import os
import base64
import requests
from io import StringIO
import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

logo_url = 'detection_logo.png'
st.title('DetectionApp')
st.sidebar.image(logo_url)
user_menu = st.sidebar.radio('Select a page', ('Predict','About our project', 'About us'))

if user_menu == 'Predict':
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

        api_url = "https://detection2-5henc2c6ta-ew.a.run.app/predict"

        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            if response.json()['status'] == 'fail':
                st.error("Failed to transform image: status='fail'.")
            else:
                st.success('Image sent successfully.')
                st.markdown(f"#### Class: {response.json()['class']}")

                proba = pd.DataFrame(response.json()['probability'])

                df = pd.DataFrame(proba.T.values,
                                columns=['Covid-19', 'Normal', 'Opacity', 'Pneumonia'],
                                index = ['Probability'])

                st.write(df)

                fig, ax = plt.subplots(1,1,figsize=(2, 2))
                ax.bar(list(df.keys()), list(df.values[0]))
                ax.tick_params(labelsize=5)
                st.pyplot(fig=fig, use_container_width=False)
        else:
            st.error("Failed to send the image. Try again later.")

if user_menu == 'About our project':
    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()

    text = read_markdown_file('README.md')
    st.markdown(text)

if user_menu == 'About us':

    st.markdown("#### Components:")

    col1,col2, col3, col4 = st.columns(4)
    with col1:
        st.write('teste')
        st.image('pictures/thaina_castro.jpg', width=100)
    with col2:
        st.image('pictures/victor_ferreira.jpeg', width=100)

    for names in os.listdir('pictures'):
        name_list = names[:names.find('.')].split('_')
        name_lastname = f"{name_list[0].capitalize()} {name_list[1].capitalize()}"
        st.image(f"pictures/{names}", width=100, caption=f"{name_lastname}")
