import streamlit as st
import pandas as pd
import subprocess
import os
from pathlib import Path 
from Pipeline import Audio_to_text, VoiceParameterEvaluator

# Define length_of_prompt as a global variable
global length_of_prompt
length_of_prompt = 1

# Create a Streamlit file uploader widget
input_type = st.selectbox('Select input type', ['Audio file', 'Plain text', 'Microphone'])

if input_type == 'Audio file':
    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])
    # If the user uploads an audio file, call the Pipeline.py script with the file path as an argument
    if audio_file is not None:
        # Get number of parameters to check from user
        length_of_prompt = st.number_input('How many parameters do you want to check?', min_value=1, max_value=15, value=1)

        # Save the uploaded file to disk
        # with open("audio_file.wav", "wb") as f:
        #     f.write(audio_file.getbuffer())

        # audio_path = r"audio_file.wav"
        # app_dir = os.path.dirname(os.path.abspath(__file__))
        # audio_path = os.path.join(app_dir, 'audio_file.wav')
        # print("audio path:",audio_path)
        # print("audio path2:",r"audio_file.wav")
        # create an instance of the Audio_to_text class and get the output DataFrame
        obj = Audio_to_text(audio_file= audio_file)
        df = obj.response_prompt(temperature=0.7,length_of_prompt=length_of_prompt)

        # # create an instance of the VoiceParameterEvaluator class and call its assign_score method
        score_obj = VoiceParameterEvaluator()
        score_obj.assign_score(df, audio_file.name + '_1.xlsx') 

        # # Call the Pipeline.py script with the file path as an argument
        # # result = subprocess.check_output(['python', 'Pipeline.py', '--audio_file', 'audio_file.wav'])

        # Get the directory of the Streamlit app file
        # app_dir = os.path.dirname(os.path.abspath(__file__))

        # # Construct the path to the Pipeline.py file
        # pipeline_path = os.path.join(app_dir, 'Pipeline.py')
        # try:
        # # Call the Pipeline.py file
        #     result = subprocess.check_output(['python', pipeline_path])
        #     print("type of result:", result)
        # except Exception as e:
        #     print("exception:", e)
        # st.write(result.decode('utf-8'))

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


# # Load the output Excel file and display it in the Streamlit app
# st.write('Output:')
# df = pd.read_excel(audio_file)
# st.write(df)