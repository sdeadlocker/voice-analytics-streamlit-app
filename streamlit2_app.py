import streamlit as st
import pandas as pd
import subprocess
import os
from pathlib import Path 
from Pipeline import Audio_to_text, VoiceParameterEvaluator

# Define length_of_prompt as a global variable
# global length_of_prompt
# length_of_prompt = 1

# Create a Streamlit file uploader widget
input_type = st.selectbox('Select input type', ['Audio file', 'Plain text', 'Microphone'])

if input_type == 'Audio file':
    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])
    
    # Use session_state to keep track of whether the user has pressed enter or not
    if 'enter_pressed' not in st.session_state:
        st.session_state.enter_pressed = False
    
    # If the user uploads an audio file, call the Pipeline.py script with the file path as an argument
    if audio_file is not None:
        # Get number of parameters to check from user
        if not st.session_state.enter_pressed:
            length_of_prompt = st.number_input('How many parameters do you want to check?', min_value=1, max_value=15, value=1, key='length_of_prompt')
            if st.button('Enter'):
                st.session_state.enter_pressed = True
        else:
            st.text(f"Number of parameters: 1")
            
        # Disable the number_input widget once the user has pressed enter
        if st.session_state.enter_pressed:
            st.markdown("<style>div[data-baseweb='input'] input {pointer-events: none;}</style>", unsafe_allow_html=True)
            
        obj = Audio_to_text(audio_file= audio_file)
        df = obj.response_prompt(temperature=0.7,length_of_prompt=length_of_prompt)

        # # create an instance of the VoiceParameterEvaluator class and call its assign_score method
        score_obj = VoiceParameterEvaluator()
        score_obj.assign_score(df, audio_file.name + '_1.xlsx') 


elif input_type == 'Plain text':
    text_input = st.text_input('Enter text')
    # If the user enters plain text, call the Pipeline.py script with the text as an argument
    if text_input:
        # Call the Pipeline.py script with the text as an argument
        result = subprocess.check_output(['python', 'Pipeline.py', '--text', text_input])
        st.write(result.decode('utf-8'))

elif input_type == 'Microphone':
    # Call the Pipeline.py script with the microphone as an argument
    result = subprocess.check_output(['python', 'Pipeline.py', '--mic'])
    st.write(result.decode('utf-8'))
