"""
Model utilities for the Speech Emotion Recognition project.

This module contains functions for loading, saving, and using the 
emotion recognition model.
"""

import os
import numpy as np
import tensorflow as tf
from .audio_processing import extract_mfcc_from_file

# Dictionary mapping model predictions to emotion labels
EMOTIONS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear', 
    3: 'happy', 
    4: 'neutral', 
    5: 'ps',  # Likely "Pleasant Surprise"
    6: 'sad'
}

# Cache for loaded models to avoid reloading
MODEL_CACHE = {}

def find_model_file(model_name='my_model.keras', search_dirs=None):
    """
    Find a model file in various directories.
    
    Args:
        model_name (str): Name of the model file
        search_dirs (list): List of directories to search in
        
    Returns:
        str: Path to the found model file, or None if not found
    """
    if search_dirs is None:
        # Current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Default search directories
        search_dirs = [
            # Models directory (next to src)
            os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'models'),
            # Current directory's parent
            os.path.dirname(current_dir),
            # speech-17apr directory
            os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'speech-17apr'),
            # Direct path to speech-17apr folder
            r'c:\Users\chira\OneDrive\Documents\speech-17apr[1]\speech-17apr'
        ]
    
    # Search for the model file
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
            
        model_path = os.path.join(directory, model_name)
        if os.path.exists(model_path):
            return model_path
            
    return None

def load_model(model_path=None):
    """
    Load the emotion recognition model.
    
    Args:
        model_path (str, optional): Path to the model file.
            If None, the function will try to find the model in default locations.
            
    Returns:
        tf.keras.Model: Loaded model, or None if loading fails
    """
    try:
        # If model_path is not provided, try to find the model
        if model_path is None:
            model_path = find_model_file()
            
            if model_path is None:
                raise FileNotFoundError("Model file not found in any of the search directories")
        
        # Check if model is already loaded
        if model_path in MODEL_CACHE:
            return MODEL_CACHE[model_path]
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Cache the model
        MODEL_CACHE[model_path] = model
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_emotion(model, audio_file):
    """
    Predict emotion from an audio file.
    
    Args:
        model (tf.keras.Model): Loaded emotion recognition model
        audio_file (str): Path to the audio file
        
    Returns:
        str: Predicted emotion
    """
    try:
        # Extract features
        features = extract_mfcc_from_file(audio_file)
        if features is None:
            return "error"
            
        # Prepare features for the model
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1)  # Add channel dimension
        
        # Make prediction
        predictions = model.predict(features)
        emotion_idx = np.argmax(predictions)
        
        # Return emotion label
        return EMOTIONS.get(emotion_idx, "unknown")
        
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return "error"

def get_all_emotions():
    """
    Get the list of all possible emotions.
    
    Returns:
        list: List of emotion labels
    """
    return list(EMOTIONS.values())

def get_model_summary(model=None):
    """
    Get a string summary of the model architecture.
    
    Args:
        model (tf.keras.Model, optional): Model to summarize.
            If None, the function will try to load the model.
            
    Returns:
        str: Model summary as string
    """
    if model is None:
        model = load_model()
        
    if model is None:
        return "Model could not be loaded"
        
    # Create a string stream to capture summary
    import io
    summary_stream = io.StringIO()
    
    # Print summary to string stream
    model.summary(print_fn=lambda x: summary_stream.write(x + '\n'))
    
    # Return summary as string
    return summary_stream.getvalue()
