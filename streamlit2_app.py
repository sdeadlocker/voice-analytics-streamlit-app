import streamlit as st
import pandas as pd
import subprocess
import os
os.chdir(os.path.dirname(__file__))
from pathlib import Path 
from Pipeline import Audio_to_text, VoiceParameterEvaluator
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf

st.set_page_config(page_title="Voice Analytics App")

st.title("Voice Analytics App")
st.header('The app allows users to input sales call data via three options, transcribe it, and generate scores for various call parameters.')

st.write('Please select an input source:')
st.write('Upload a audio/Record a audio/Write text to get started')

# Create a Streamlit file uploader widget
input_type = st.selectbox('Select input type', ['Audio file', 'Microphone', 'Plain text'])

# Create a "Stored Data" folder if it doesn't exist
if not os.path.exists("Stored Data"):
    os.makedirs("Stored Data")
if input_type == 'Audio file':

    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])
    
    # Use session_state to keep track of whether the user has pressed enter or not
    if 'enter_pressed' not in st.session_state:
        st.session_state.enter_pressed = False
    
    # If the user uploads an audio file, call the Pipeline.py script with the file path as an argument
    if audio_file is not None:
        # Save the uploaded audio file in the "Stored Data" folder
        file_path = os.path.join("Stored Data", audio_file.name)
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Get number of parameters to check from user
        if not st.session_state.enter_pressed:
            length_of_prompt = st.number_input('How many parameters do you want to check?', min_value=1, max_value=15, value=15, key='length_of_prompt')
            if st.button('Enter'):
                st.session_state.enter_pressed = True
        else:
            st.text(f"Number of parameters: 15")
            
        # Disable the number_input widget once the user has pressed enter
        if st.session_state.enter_pressed:
            st.markdown("<style>div[data-baseweb='input'] input {pointer-events: none;}</style>", unsafe_allow_html=True)
            
        obj = Audio_to_text(audio_file= audio_file)
        df = obj.response_prompt(temperature=0.7,length_of_prompt=length_of_prompt)

        # create an instance of the VoiceParameterEvaluator class and call its assign_score method
        score_obj = VoiceParameterEvaluator()
        output_file_path = os.path.join("Stored Data", audio_file.name + '_1.xlsx')
        score_obj.assign_score(df, output_file_path) 

        # Read the output Excel file into a pandas dataframe
        df_output = pd.read_excel(output_file_path)
        # Apply automatic type conversion to make the dataframe Arrow-compatible
        try:
            df_output = df_output.convert_dtypes()
        except:
            st.warning("Automatic type conversion was unsuccessful. Please check your data types.")

        st.text('Output file generated')
        # Display the output dataframe in the Streamlit app
        st.write(df_output.T)


# If the user selects the microphone, record audio and save it to disk, then call the Pipeline
elif input_type == 'Microphone':
    st.write('Click the button below to start recording:')
    record_button = st.button('Record', key='record_button')
    stop_button = st.button('Stop Recording',key='stop_button')
    # submit_button = st.button('Submit',key='submit_button')

# Define a function to record audio
    def record_audio():
        freq  = 44100  # Sample rate
        seconds = 10  # Duration of recording
        print("Recording started...")
        # Record audio
        recording = sd.rec(int(seconds * freq), samplerate=freq, channels=2)
        sd.wait()
        print("Recording stopped!")
        # Set path for saving audio
        audio_path = os.path.join(os.getcwd(), "Stored Data", "audio.wav")
        # Save audio file
        write(audio_path, freq, recording)
        print(f"Audio saved at {audio_path}")
        return audio_path

    audio_path = record_audio()
    # Use session_state to keep track of whether the user has started recording or not
    if 'recording' not in st.session_state:
        st.session_state.recording = False

     # If the user clicks the record button, start recording audio
    if record_button and not st.session_state.recording:
        st.session_state.recording = True
        st.text('Recording started...')
        st.text('(Please speak into your microphone)')
        st.text('(Click "Stop Recording" to finish and transcribe)')

    # If the user clicks the stop button, stop recording audio
    if stop_button and st.session_state.recording:
        st.session_state.recording = False
        st.text('Recording stopped!')
        audio_path = record_audio()
        st.text(f'Recorded audio saved to {audio_path}')
    
    length_of_prompt = st.number_input('How many parameters do you want to check?', min_value=1, max_value=15, value=15, key='length_of_prompt')
  
    submit_button = st.button('Submit',key='submit_button')
    
    # If the user clicks the submit button, transcribe the recorded audio and show the output
    if submit_button and not st.session_state.recording:
        st.text('Transcribing audio...')
        # Get the file path of the recorded audio
        audio_path = os.path.join(os.getcwd(), 'Stored Data', 'audio.wav')
        print('audio path:',audio_path)
        obj = Audio_to_text(audio_file=open(audio_path, 'rb'))
        df = obj.response_prompt(temperature=0.7, length_of_prompt=length_of_prompt)
        score_obj = VoiceParameterEvaluator()
        output_file_path = os.path.join(os.getcwd(), 'Stored Data', 'recorded_audio.xlsx')
        score_obj.assign_score(df, output_file_path)
        # st.text('Output file generated')
        # st.text(f'You can find the output file at {os.getcwd()}/recorded_audio_1.xlsx')

        # Read the output Excel file into a pandas dataframe
        df_output = pd.read_excel(output_file_path)
        st.text('Output file generated')
        # Display the output dataframe in the Streamlit app
        st.write(df_output.T)

elif input_type == 'Plain text':
    text_input = st.text_input('Enter text')
    # If the user enters plain text, call the Pipeline.py script with the text as an argument
    if text_input:
        # Call the Pipeline.py script with the text as an argument
        result = subprocess.check_output(['python', 'Pipeline.py', '--text', text_input])
        st.write(result.decode('utf-8'))