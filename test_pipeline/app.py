import streamlit as st
# from Pipeline import 
import subprocess
import pandas as pd
import os

# Create a Streamlit file uploader widget
input_type = st.selectbox('Select input type', ['Audio file', 'Plain text', 'Video file'])

if input_type == 'Audio file':
    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])
    # If the user uploads an audio file, save it to disk and call the Pipeline.py script with the audio file path as an argument
    if audio_file is not None:
        # Save the uploaded file to disk
        audio_path = os.path.join('data', 'audio_file.wav')
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Get the directory of the Streamlit app file
        app_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the Pipeline.py file
        pipeline_path = os.path.join(app_dir, 'Pipeline.py')

        # Call the Pipeline.py script with the audio file path as an argument
        output_path = os.path.join(os.path.dirname(audio_path), 'output.xlsx')
        subprocess.run(['python', pipeline_path, '--audio_path', audio_path, '--output_path', output_path])

        # Load the output Excel file and display it in the Streamlit app
        st.write('Output:')
        df = pd.read_excel(output_path)
        st.write(df)

elif input_type == 'Plain text':
    text_input = st.text_input('Enter text')
    # If the user enters plain text, call the Pipeline.py script with the text as an argument
    if text_input:
        # Call the Pipeline.py script with the text as an argument
        output_path = os.path.join('data', 'output.xlsx')
        subprocess.run(['python', 'Pipeline.py', '--text', text_input, '--output_path', output_path])

        # Load the output Excel file and display it in the Streamlit app
        st.write('Output:')
        df = pd.read_excel(output_path)
        st.write(df)

elif input_type == 'Video file':
    video_file = st.file_uploader("Upload video file", type=["mp4", "avi"])
    # If the user uploads a video file, save it to disk and call the Pipeline.py script with the video file path as an argument
    if video_file is not None:
        # Save the uploaded file to disk
        video_path = os.path.join('data', 'video_file.mp4')
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Get the directory of the Streamlit app file
        app_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the Pipeline.py file
        pipeline_path = os.path.join(app_dir, 'Pipeline.py')

        # Call the Pipeline.py script with the video file path as an argument
        output_path = os.path.join(os.path.dirname(video_path), 'output.xlsx')
        subprocess.run(['python', pipeline_path, '--video_path', video_path, '--output_path', output_path])

        # Load the output Excel file and display it in the Streamlit app
        st.write('Output:')
        df = pd.read_excel(output_path)
        st.write(df)
