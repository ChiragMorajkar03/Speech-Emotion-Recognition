"""
Speech Emotion Recognition Web Application

A Streamlit web app that allows users to upload or record audio
and predicts the emotion in the speech.

Author: Chirag Morajkar
Date: June 2025
"""

import os
import sys
import tempfile
import streamlit as st
import sounddevice as sd
import soundfile as sf
import tensorflow as tf

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import our custom modules
from src.audio_processing import apply_gaussian_filter, load_audio, save_audio
from src.model_utils import load_model, predict_emotion, get_all_emotions, get_model_summary

# Constants
SAMPLE_RATE = 44100
RECORDING_DURATION = 3  # seconds
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'speech_emotion_recognition')

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'emotion_colors' not in st.session_state:
        st.session_state.emotion_colors = {
            'angry': '#FF5252',     # Red
            'disgust': '#8BC34A',   # Light Green
            'fear': '#9C27B0',      # Purple
            'happy': '#FFD600',     # Yellow
            'neutral': '#03A9F4',   # Light Blue
            'ps': '#FF9800',        # Orange
            'sad': '#607D8B',       # Blue Gray
            'unknown': '#9E9E9E',   # Gray
            'error': '#F44336'      # Error Red
        }

def record_audio():
    """Record audio from the microphone."""
    try:
        st.info(f"Recording started. Please speak for {RECORDING_DURATION} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(RECORDING_DURATION * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='float64'
        )
        
        # Wait for recording to complete
        sd.wait()
        
        # Generate a unique filename for this recording
        filename = os.path.join(TEMP_DIR, f'recording_{tempfile.NamedTemporaryFile().name}.wav')
        
        # Save the recording
        sf.write(filename, recording, SAMPLE_RATE)
        
        return filename
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    """Process an uploaded audio file."""
    try:
        # Generate a temporary filename
        temp_path = os.path.join(TEMP_DIR, f"uploaded_{tempfile.NamedTemporaryFile().name}.wav")
        
        # Save the uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        return temp_path
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

def apply_filtering(audio_path):
    """Apply audio filtering to improve quality."""
    try:
        # Load audio
        audio_data, sampling_rate = load_audio(audio_path)
        
        if audio_data is None:
            return None
            
        # Apply Gaussian filter
        filtered_audio = apply_gaussian_filter(audio_data)
        
        # Save filtered audio
        filtered_path = os.path.join(TEMP_DIR, f"filtered_{os.path.basename(audio_path)}")
        save_audio(filtered_path, filtered_audio, sampling_rate)
        
        return filtered_path
    except Exception as e:
        st.error(f"Error applying audio filter: {str(e)}")
        return None

def display_emotion(emotion):
    """Display the detected emotion with appropriate styling."""
    color = st.session_state.emotion_colors.get(emotion, '#9E9E9E')
    
    # Map emotions to emoji
    emotion_emojis = {
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😄',
        'neutral': '😐',
        'ps': '😲',  # Pleasant surprise
        'sad': '😢',
        'unknown': '❓',
        'error': '⚠️'
    }
    
    emoji = emotion_emojis.get(emotion, '❓')
    
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {color}33; border: 2px solid {color};">
            <h2 style="text-align: center; color: {color}; margin: 0;">
                {emoji} {emotion.upper()} {emoji}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def cleanup_temp_files():
    """Clean up temporary files older than 1 hour."""
    import time
    
    current_time = time.time()
    one_hour_in_seconds = 3600
    
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            
            # Check if file is older than 1 hour
            if os.path.isfile(file_path) and (current_time - os.path.getmtime(file_path)) > one_hour_in_seconds:
                try:
                    os.unlink(file_path)
                except:
                    pass
    except:
        # Silently fail if cleanup encounters issues
        pass

def display_audio_waveform(audio_path):
    """Display an audio waveform visualization."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Load audio
        audio_data, sampling_rate = load_audio(audio_path)
        
        if audio_data is None:
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Time array
        time = np.arange(0, len(audio_data)) / sampling_rate
        
        # Plot waveform
        ax.plot(time, audio_data, color='#1DB954')
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Display the plot
        st.pyplot(fig)
    except:
        # Silently fail if visualization encounters issues
        pass

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    st.title('🎤 Speech Emotion Recognition')
    
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 3em;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        This application detects emotions in speech using deep learning.
        Upload an audio file or record your voice to analyze the emotion.
    """)
    
    # Load model
    with st.spinner('Loading emotion recognition model...'):
        model = load_model()
        if model is None:
            st.error("❌ Failed to load model. Please check if model files exist.")
            st.info("Expected location: './models/my_model.keras' or './speech-17apr/my_model.keras'")
            return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload Audio", "🎙️ Record Audio"])
    
    # Tab 1: Upload audio
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload an audio file", 
            type=["wav", "mp3", "ogg"]
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
            
            if st.button('Analyze Uploaded Audio', key='analyze_upload'):
                with st.spinner('Processing audio...'):
                    # Save uploaded file
                    audio_path = process_uploaded_file(uploaded_file)
                    
                    if audio_path:
                        # Apply filtering
                        filtered_path = apply_filtering(audio_path)
                        
                        if filtered_path:
                            st.subheader("Filtered Audio")
                            st.audio(filtered_path, format='audio/wav')
                            
                            # Display waveform
                            display_audio_waveform(filtered_path)
                            
                            # Predict emotion
                            with st.spinner('Analyzing emotion...'):
                                emotion = predict_emotion(model, filtered_path)
                                
                                # Display result
                                st.subheader("Emotion Analysis Result")
                                display_emotion(emotion)
    
    # Tab 2: Record audio
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('🎙️ Start Recording', key='record_btn'):
                with st.spinner('Recording audio...'):
                    recorded_audio_path = record_audio()
                    
                    if recorded_audio_path:
                        # Set session state
                        st.session_state.recorded_audio_path = recorded_audio_path
                        st.experimental_rerun()
        
        with col2:
            if st.button('🗑️ Clear Recording', key='clear_btn'):
                # Clear recorded audio if any
                if 'recorded_audio_path' in st.session_state:
                    try:
                        os.unlink(st.session_state.recorded_audio_path)
                    except:
                        pass
                    del st.session_state.recorded_audio_path
                    st.experimental_rerun()
        
        # Display and analyze recorded audio if available
        if 'recorded_audio_path' in st.session_state:
            audio_path = st.session_state.recorded_audio_path
            
            if os.path.exists(audio_path):
                st.subheader("Recorded Audio")
                st.audio(audio_path, format='audio/wav')
                
                if st.button('Analyze Recording', key='analyze_recording'):
                    # Apply filtering
                    filtered_path = apply_filtering(audio_path)
                    
                    if filtered_path:
                        st.subheader("Filtered Audio")
                        st.audio(filtered_path, format='audio/wav')
                        
                        # Display waveform
                        display_audio_waveform(filtered_path)
                        
                        # Predict emotion
                        with st.spinner('Analyzing emotion...'):
                            emotion = predict_emotion(model, filtered_path)
                            
                            # Display result
                            st.subheader("Emotion Analysis Result")
                            display_emotion(emotion)
    
    # Add an about section
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        ### Speech Emotion Recognition
        
        This application uses a deep learning model trained on speech datasets to recognize emotions.
        
        **How it works:**
        1. Audio is processed to extract Mel-frequency cepstral coefficients (MFCCs)
        2. These features are passed to a neural network model
        3. The model classifies the emotion in the speech
        
        **Emotions detected:**
        - Angry
        - Disgust
        - Fear
        - Happy
        - Neutral
        - Pleasant Surprise
        - Sad
        
        **Technologies used:**
        - TensorFlow for the deep learning model
        - Librosa for audio processing
        - Streamlit for the web interface
        
        Source code available on [GitHub](https://github.com/yourusername/speech-emotion-recognition)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Chirag Morajkar | Speech Emotion Recognition")
    
    # Cleanup old temp files
    cleanup_temp_files()

if __name__ == "__main__":
    main()
