import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import os
import sounddevice as sd
import tempfile
from scipy.signal import gaussian, convolve

def main():
    st.title('**Emotion Recognition**')  # Display title in bold
    application()

# Initialize an empty dictionary for caching
cached_data = {}

def load_model():
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full file path to the model file
    model_path = os.path.join(current_dir, 'my_model.keras')

    # Load the model using the full file path
    if model_path not in cached_data:
        model = tf.keras.models.load_model(model_path)
        cached_data[model_path] = model
    return cached_data[model_path]


def get_file_content_as_string(path):
    if path not in cached_data:
        with open(path, 'r') as file:
            cached_data[path] = file.read()
    return cached_data[path]

def application():
    from scipy.ndimage import gaussian_filter1d  # Import gaussian_filter1d here

    models_load_state = st.text('Loading Models ...')
    model = load_model()
    models_load_state.text('Models Loading ... complete')
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    
    # Use a session state to keep track of recording state across requests
    recording_state = st.session_state.get('recording_state', False)
    if st.button('Start Recording'):
        if recording_state:  # If already recording, stop recording
            recording_state = False
            st.session_state['recording_state'] = recording_state
            st.experimental_rerun()  # Rerun the app to stop recording
        else:  # If not recording, start recording
            recording_state = True
            st.session_state['recording_state'] = recording_state
    if recording_state:
        st.warning("Recording in progress... Click 'Start Recording' again to stop.")
    elif not recording_state and 'recording_state' in st.session_state:
        st.success("Recording stopped.")  # Display only when recording is stopped

    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Apply Gaussian filter to the uploaded audio
        audio_data, fs = sf.read("uploaded_audio.wav")
        audio_data_filtered = gaussian_filter1d(audio_data, sigma=2)
        sf.write("uploaded_audio_filtered.wav", audio_data_filtered, fs)
        
        st.audio("uploaded_audio_filtered.wav", format='audio/wav')
        st.success('Emotion of the Audio is ' + predict(model, "uploaded_audio_filtered.wav"))
    
    elif recording_state:
        audio_file_path = record_audio()
        st.audio(audio_file_path, format='audio/wav')
        st.success('Emotion of the Audio is ' + predict(model, audio_file_path))

def extract_mfcc(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict(model, wav_file):
    emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
    test_point = extract_mfcc(wav_file)
    test_point = np.expand_dims(test_point, axis=0)
    test_point = np.expand_dims(test_point, axis=-1)
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions)]

def record_audio():
    duration = 3  # seconds
    fs = 44100
    
    st.info("Recording started. Please speak now...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    sf.write(temp_path, myrecording, fs)
    
    # Apply Gaussian filter to the recorded audio
    audio_data, _ = sf.read(temp_path)
    gaussian_window = gaussian(9, std=2)  # Gaussian window with std=2
    audio_data_filtered = convolve(audio_data, gaussian_window, mode='same') / gaussian_window.sum()
    sf.write(temp_path, audio_data_filtered, fs)
    
    return temp_path

if __name__ == "__main__":
    main()
