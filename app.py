import streamlit as st
import pandas as pd
import subprocess
import os
os.chdir(os.path.dirname(__file__))
from pathlib import Path 
from Pipeline import Audio_to_text, VoiceParameterEvaluator
from Text_Pipeline import Plain_text, PlainTextParameterEvaluator
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime

file=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

# import soundfile as sf
disabled_val = True
st.set_page_config(page_title="Voice Analytics App", initial_sidebar_state="collapsed", layout="wide")

# Define custom CSS to position the header and adjust margins
custom_css = """
<style>
header {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100px;
    margin: 0;
    margin-top: 0px;
    padding: 0 50px;
    background-color: #F5F5F5;
    box-shadow: none;
}
header h1 {
    margin: 0;
}
body {
    background-color: #f2f2f2;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Add the main header content
header_html = """
<header>
    <h1>Voice Analytics App</h1>
</header>
"""
st.write(header_html, unsafe_allow_html=True)
# Add an image

st.image("https://www.cleartouch.in/wp-content/uploads/2022/03/BlogImage-Everything-you-wanted-to-know-about-voice-analytics.png", 
         use_column_width=True, caption="Automating sales call auditing")
# st.image("https://www.cleartouch.in/wp-content/uploads/2022/03/BlogImage-Everything-you-wanted-to-know-about-voice-analytics.png", 
#          width=1000 , caption="Automating sales call auditing")

# Add the heading content
# st.header('The app allows users to input sales call data via three options')
st.markdown("### This app allows users to input sales call data via three options(Audio File/Microphone/Plain Text)")
# st.markdown("### Depending upon input, it will generate the scores for various sales call parameters.")

# st.write('Please select an input source:')
st.markdown('**Please select an input source:**')
st.write('Upload a audio/Record a audio/Write text to get started')

input_type = st.selectbox('Select input type', ['Audio file', 'Microphone', 'Plain text'])
st.markdown("""
<style>
.css-1iyw2u1 {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

def disable():
    disabled_val = True
    #   st.session_state["disabled"] = True
    

# Create a "Stored Data" folder if it doesn't exist
if not os.path.exists("Stored Data"):
    os.makedirs("Stored Data")

if input_type == 'Audio file':
    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

    # Use session_state to keep track of whether the user has pressed enter or not
    if 'enter_pressed' not in st.session_state:
        st.session_state.enter_pressed = False

    # If the user uploads an audio file, display the number of parameters to check and prompt the user to press enter
    if audio_file is not None:
        # Save the uploaded audio file in the "Stored Data" folder
        file_path = os.path.join("Stored Data", audio_file.name)
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Display the number of parameters to check and prompt the user to press enter
        st.markdown("#### Number of parameters we are checking is 26")
        if not st.session_state.enter_pressed:
            st.text("Please press the 'Enter' button to generate results.")
            
            st.session_state.disabled = False
            Enter_sub= st.empty()
            enter_btn = Enter_sub.button('Enter',disabled=st.session_state.disabled) 
            # print("before",disabled_val)
            # if st.button('Enter'):
            #     st.session_state.enter_pressed = True
            #     # Disable all the input widgets
            #     if 'length_of_prompt' in st.session_state:
            #         st.session_state.input_disabled = True
        else:
            st.text(f"Number of parameters: 26")

        # Disable the input widgets once the user has pressed enter
        if st.session_state== True:
            st.session_state.input_disabled = True
        
   
         
        
        # if st.button('Submit',disabled=disabled_val,on_click=disable):
        
        if enter_btn:
            
            st.session_state.disabled = True
            # st.write('You have selected ', submit_btn)
            Enter_btn = Enter_sub.button('Processing',disabled=st.session_state.disabled)
            # st.snow()
            # disabled=st.session_state.disabled
            # print("after",disabled_val)

        
        length_of_prompt = 15

        obj = Audio_to_text(audio_file=audio_file)
        df = obj.response_prompt(temperature=0.7, length_of_prompt=length_of_prompt)

        # create an instance of the VoiceParameterEvaluator class and call its assign_score method
        score_obj = VoiceParameterEvaluator()
        output_file_path = os.path.join("Stored Data", audio_file.name + '_1.xlsx')
        score_obj.assign_score(df, output_file_path)

        # Read the output Excel file into a pandas dataframe
        df_output = pd.read_excel(output_file_path)
        # Apply automatic type conversion to make the dataframe Arrow-compatible
        try:
            # df_output = df_output.convert_dtypes()
            df_output = df_output.astype(str)

        except:
            st.warning("Automatic type conversion was unsuccessful. Please check your data types.")
        
        df_output.insert(0, 'Parameters', '')

        # set first row of 'Parameters' column to 'Scores and Reason'
        df_output.at[0, 'Parameters'] = 'Scores and Reason'

        # Transpose the DataFrame and reset index to column names
        df_output = df_output.T.reset_index()
        # Set the column names to the first row and drop the old index column
        df_output.columns = df_output.iloc[0]
        df_output = df_output.drop(0)
        # Display the output dataframe in the Streamlit app
        st.text('Output file generated')
        st.write(df_output)

        # Enable the input widgets after displaying the output
        if 'input_disabled' in st.session_state:
            st.session_state.input_disabled = False

        # Reset enter_pressed session state variable so user can enter new number of parameters for the next file
        st.session_state.enter_pressed = False


# If the user selects the microphone, record audio and save it to disk, then call the Pipeline
elif input_type == 'Microphone':
    st.write('Click the button below to start recording:')
    # record_button = st.button('Record', key='record_button')
    # stop_button = st.button('Stop Recording',key='stop_button')
    
    # For record button
    st.session_state.disabled = False
    record= st.empty()
    record_button= record.button('Record',disabled=st.session_state.disabled) 
    # print("before",disabled_val)
    
    #For stopping the recording
    st.session_state.disabled = False
    stop= st.empty()
    stop_button = stop.button('Stop Recording',disabled=st.session_state.disabled) 
    # print("before",disabled_val)
    

# Define a function to record audio
    def record_audio():
        freq  = 44100  # Sample rate
        seconds = 1000 # Duration of recording
        # print("Recording started...")
        # Record audio
        recording = sd.rec(int(seconds * freq), samplerate=freq, channels=1)
        sd.wait()
        # Set path for saving audio
        audio_path = os.path.join(os.getcwd(), "Stored Data", "audio.wav")
        # Save audio file
        write(audio_path, freq, recording)
        print(f"Audio saved at {audio_path}")
        return audio_path
    
    customized_button = st.markdown("""
        <style >
        .strecord_button, div.stButton {text-align:center}
        </style>""", unsafe_allow_html=True)
 
    def output_dataframe(audio_path):
        
         # Get the file path of the recorded audio
        audio_path = os.path.join(os.getcwd(), 'Stored Data', 'audio.wav')
        print('audio path:', audio_path)
        obj = Audio_to_text(audio_file=open( audio_path, 'rb'))
        df = obj.response_prompt(temperature=0.7, length_of_prompt=15)
        score_obj = VoiceParameterEvaluator()
        output_file_path = os.path.join(os.getcwd(), 'Stored Data', 'recorded_audio.xlsx')
        score_obj.assign_score(df, output_file_path)

        # Read the output Excel file into a pandas dataframe
        df_output = pd.read_excel(output_file_path)
         # create new column 'Parameters' with blank values
        df_output.insert(0, 'Parameters', '')
        
        # set first row of 'Parameters' column to 'Scores and Reason'
        df_output.at[0, 'Parameters'] = 'Scores and Reason'

        # Transpose the DataFrame and reset index to column names
        df_output = df_output.T.reset_index()
        # Set the column names to the first row and drop the old index column
        df_output.columns = df_output.iloc[0]
        df_output = df_output.drop(0)
        
        st.markdown("### Output file generated")
        # Display the output dataframe in the Streamlit app
        st.write(df_output)
        
    # Use session_state to keep track of whether the user has started recording or not
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    audio_path = ''
    
    
     # If the user clicks the record button, start recording audio
    if record_button and not st.session_state.recording:
        
        st.session_state.recording = True
        st.session_state.disabled = True
        # st.write('You have selected ', submit_btn)
        record_button = record.button('Recording',disabled=st.session_state.disabled)
 
        # st.snow()
        # disabled=st.session_state.disabled
        # print("after",disabled_val)
        st.text('Recording started...')
        st.text('(Please speak into your microphone)')
        st.text('(Click "Stop Recording" to finish and transcribe)')
        audio_path = record_audio()
        output_dataframe(audio_path)
    else:
        st.text('')    

    # If the user clicks the stop button, stop recording audio and generate results
    if stop_button and st.session_state.recording:
        st.session_state.recording = False
        st.text('Recording stopped!')
        st.markdown("#### Number of parameters we are checking is 26")
        st.session_state.disabled = True
        # st.write('You have selected ', submit_btn)
        stop_button = stop.button('Recording Stopped',disabled=st.session_state.disabled)
        # st.snow()
       
        st.text('Generating results...')
        output_dataframe(audio_path)
        
        
        customized_button = st.markdown("""
        <style >
        .strecord_button, div.stButton {text-align:center}            }
        </style>""", unsafe_allow_html=True)

elif input_type == 'Plain text':
   
     # Use session_state to keep track of whether the user has pressed enter or not
    if 'enter_pressed' not in st.session_state:
        st.session_state.enter_pressed = False
        
    if "disabled" not in st.session_state:
        st.session_state.disabled = False
  
    # Create a text box for the user to enter their text
    input_text = st.text_area("Enter your text here:", value="",height=250)
        
   # Display the number of parameters to check
    st.text("Number of parameters we are checking is 26")

 
    st.session_state.disabled = False
    submit= st.empty()
    submit_btn = submit.button('Submit',disabled=st.session_state.disabled) 
    # print("before",disabled_val)
    # if st.button('Submit',disabled=disabled_val,on_click=disable):
    if submit_btn:
        
        st.session_state.disabled = True
        # st.write('You have selected ', submit_btn)
        submit_btn = submit.button('Submiting',disabled=st.session_state.disabled)
        # st.snow()
        # disabled=st.session_state.disabled
        # print("after",disabled_val)
     
        
        # Save the input text to a file
        file_path = os.path.join("Stored Data", "Text"+ -(file)+".txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(input_text)
            
        # Set the default number of parameters to check to 15
        length_of_prompt = 15


        # Call the PlainText class from Text_Input.py and pass the file path as an argument
        obj = Plain_text(input_text=input_text)
        df = obj.response_prompt(temperature=0.7, length_of_prompt=length_of_prompt)

        # create an instance of the PlainTextParameterEvaluator class and call its assign_score method
        score_obj = PlainTextParameterEvaluator()
        output_file_path = os.path.join("Stored Data", "Text-"+(file)+".xlsx")
        score_obj.assign_score(df, output_file_path)

        # Read the output Excel file into a pandas dataframe
        df_output = pd.read_excel(output_file_path)
        # Apply automatic type conversion to make the dataframe Arrow-compatible
        try:
            # df_output = df_output.convert_dtypes()
            df_output = df_output.astype(str)
            # df_output1[['file_name','Conversation','Score','Reason']]=df_output[]

        except:
            st.warning("Automatic type conversion was unsuccessful. Please check your data types.")
        
        
         # create new column 'Parameters' with blank values
        df_output.insert(0, 'Parameters', '')

        # set first row of 'Parameters' column to 'Scores and Reason'
        df_output.at[0, 'Parameters'] = 'Scores and Reason'

        # Transpose the DataFrame and reset index to column names
        df_output = df_output.T.reset_index()

        # Set the column names to the first row and drop the old index column
        df_output.columns = df_output.iloc[0]
        df_output = df_output.drop(0)

        # Display the output dataframe in the Streamlit app
        # st.text('Output file generated')
        st.markdown("### Output file generated")
        st.write(df_output)
        # Clear the text box content
        
    

    # Reset enter_pressed session state variable so user can enter new number of parameters for the next input
    st.session_state.enter_pressed = False
