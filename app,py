import streamlit as st
import pandas as pd
import numpy as np
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg_5hands.jpg')

def main():
    st.header(':gray[_Welcome to NyoKi Classifier_]')
    st.subheader('Hand Sign Recognition Application (Word Level)')
    activities = ["Home", "Webcam Hand Detection", "Thanks"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "home":
         html_temp_home1 = """<div style="background-color:#454545;padding:10px">
                              <h4 style="color:white;text-align:center;">
                              Hand Sign recognition application using OpenCV, Streamlit.
                              </h4>
                              </div>
                              </br>"""
         st.markdown(html_temp_home1, unsafe_allow_html=True)
#elif choice == "Webcam Hand Detection":

    
                              
                              
                              
                              
                              
                              
                              
                              
