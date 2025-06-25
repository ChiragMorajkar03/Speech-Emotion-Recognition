"""
Speech Emotion Recognition Application

This application uses a trained deep learning model to recognize emotions from speech audio.
It provides a web interface using Streamlit for users to upload audio files or record live audio.

Author: Chirag Morajkar
Date: June 2025
"""

import os
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
import librosa
import soundfile as sf
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import gaussian, convolve

# Constants
SAMPLE_RATE = 44100
RECORDING_DURATION = 3  # seconds
MFCC_COUNT = 40
EMOTIONS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear', 
    3: 'happy', 
    4: 'neutral', 
    5: 'ps', 
    6: 'sad'
}

# Initialize cache dictionary
MODEL_CACHE = {}

def load_model():
    """
    Load the trained emotion recognition model with caching for performance.
    
    Returns:
        model: Loaded TensorFlow model
    """
    try:
        # Get the absolute path of the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to model directory (parent directory/models)
        parent_dir = os.path.dirname(current_dir)
        model_dir = os.path.join(parent_dir, 'models')
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            st.warning(f"Created models directory at {model_dir}")
            
        # Try to find model file in models directory first
        model_path = os.path.join(model_dir, 'my_model.keras')
        
        # If model not in models directory, try current directory
        if not os.path.exists(model_path):
            model_path = os.path.join(current_dir, 'my_model.keras')
            
            # If model exists in current directory but not in models directory, copy it
            if os.path.exists(model_path):
                st.warning("Model found in current directory. Consider moving it to models/ directory.")
        
        # Load the model with caching
        if model_path not in MODEL_CACHE:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path)
            
        return MODEL_CACHE[model_path]
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_mfcc(audio_file):
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from an audio file.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        mfccs: Numpy array of MFCC features
    """
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COUNT).T, axis=0)
        return mfccs
    except Exception as e:
        st.error(f"Error extracting MFCC features: {str(e)}")
        return None

def apply_gaussian_filter(audio_data, sigma=2):
    """
    Apply Gaussian filter to audio data.
    
    Args:
        audio_data: Audio data as numpy array
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        filtered_audio: Filtered audio data
    """
    return gaussian_filter1d(audio_data, sigma=sigma)

def predict_emotion(model, audio_file):
    """
    Predict emotion from audio file using the loaded model.
    
    Args:
        model: TensorFlow model for emotion prediction
        audio_file: Path to audio file
        
    Returns:
        emotion: Predicted emotion label
    """
    try:
        if model is None:
            return "Error: Model not loaded"
            
        # Extract features
        test_point = extract_mfcc(audio_file)
        if test_point is None:
            return "Error: Could not extract features"
            
        # Prepare input for model
        test_point = np.expand_dims(test_point, axis=0)
        test_point = np.expand_dims(test_point, axis=-1)
        
        # Make prediction
        predictions = model.predict(test_point)
        emotion_idx = np.argmax(predictions)
        
        return EMOTIONS.get(emotion_idx, "unknown")
        
    except Exception as e:
        return f"Error during prediction: {str(e)}"

def record_audio():
    """
    Record audio from microphone and save to temporary file.
    
    Returns:
        temp_path: Path to temporary audio file with recorded audio
    """
    try:
        # Record audio
        st.info(f"Recording started. Please speak for {RECORDING_DURATION} seconds...")
        recording = sd.rec(
            int(RECORDING_DURATION * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='float64'
        )
        sd.wait()  # Wait until recording is done
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        sf.write(temp_path, recording, SAMPLE_RATE)
        
        # Apply Gaussian filter
        audio_data, _ = sf.read(temp_path)
        filtered_audio = apply_gaussian_filter(audio_data)
        sf.write(temp_path, filtered_audio, SAMPLE_RATE)
        
        return temp_path
        
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def process_uploaded_audio(uploaded_file):
    """
    Process an uploaded audio file.
    
    Args:
        uploaded_file: UploadedFile object from Streamlit
        
    Returns:
        temp_path: Path to processed audio file
    """
    try:
        # Create temporary file for uploaded audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        
        # Save uploaded audio to temp file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Apply Gaussian filter
        audio_data, fs = sf.read(temp_path)
        filtered_audio = apply_gaussian_filter(audio_data)
        
        # Save filtered audio
        filtered_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        sf.write(filtered_path, filtered_audio, fs)
        
        # Clean up original temp file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return filtered_path
        
    except Exception as e:
        st.error(f"Error processing uploaded audio: {str(e)}")
        return None

def main():
    """Main application function."""
    # Set page configuration
    st.set_page_config(
        page_title="Speech Emotion Recognition",
        page_icon="🎤",
        layout="centered"
    )
    
    # Display header
    st.title('🎤 Speech Emotion Recognition')
    st.markdown("""
        This application detects emotions in speech using deep learning.
        Upload an audio file or record your voice to analyze the emotion.
    """)
    
    # Load model
    with st.spinner('Loading emotion recognition model...'):
        model = load_model()
        if model is None:
            st.error("Failed to load model. Please check logs.")
            return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
    
    # Tab 1: Upload audio
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload an audio file", 
            type=["wav", "mp3", "ogg"]
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
            
            with st.spinner('Processing audio...'):
                processed_audio_path = process_uploaded_audio(uploaded_file)
                
                if processed_audio_path:
                    st.subheader("Filtered Audio")
                    st.audio(processed_audio_path, format='audio/wav')
                    
                    # Predict emotion
                    with st.spinner('Analyzing emotion...'):
                        emotion = predict_emotion(model, processed_audio_path)
                        
                        # Display result
                        st.markdown(f"### Detected Emotion: **{emotion.upper()}**")
                        
                        # Clean up temp file
                        try:
                            os.unlink(processed_audio_path)
                        except:
                            pass
    
    # Tab 2: Record audio
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('Start Recording', key='record_btn'):
                with st.spinner('Recording and processing audio...'):
                    recorded_audio_path = record_audio()
                    
                    if recorded_audio_path:
                        # Set session state to store the path
                        st.session_state.recorded_audio_path = recorded_audio_path
                        st.experimental_rerun()
        
        with col2:
            if st.button('Clear Recording', key='clear_btn'):
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
                
                # Predict emotion
                with st.spinner('Analyzing emotion...'):
                    emotion = predict_emotion(model, audio_path)
                    
                    # Display result with color based on emotion
                    st.markdown(f"### Detected Emotion: **{emotion.upper()}**")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Chirag Morajkar | Speech Emotion Recognition")

if __name__ == "__main__":
    main()
