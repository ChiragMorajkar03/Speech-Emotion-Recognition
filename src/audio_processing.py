"""
Audio processing utilities for the Speech Emotion Recognition project.

This module contains functions for processing audio files, extracting features,
and applying filters for the speech emotion recognition system.
"""

import os
import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import gaussian, convolve

def load_audio(file_path, sr=None):
    """
    Load an audio file using librosa.
    
    Args:
        file_path (str): Path to the audio file
        sr (int, optional): Target sampling rate. Defaults to None (original sampling rate).
        
    Returns:
        tuple: (audio_data, sampling_rate)
    """
    try:
        audio_data, sampling_rate = librosa.load(file_path, sr=sr)
        return audio_data, sampling_rate
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        return None, None

def apply_gaussian_filter(audio_data, sigma=2):
    """
    Apply Gaussian filter to audio data.
    
    Args:
        audio_data (numpy.ndarray): Audio data array
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        numpy.ndarray: Filtered audio data
    """
    return gaussian_filter1d(audio_data, sigma=sigma)

def extract_mfcc(audio_data, sampling_rate, n_mfcc=40):
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from audio data.
    
    Args:
        audio_data (numpy.ndarray): Audio data
        sampling_rate (int): Sampling rate of audio
        n_mfcc (int): Number of MFCCs to extract
        
    Returns:
        numpy.ndarray: MFCC features
    """
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC features: {str(e)}")
        return None
        
def extract_mfcc_from_file(file_path, n_mfcc=40):
    """
    Extract MFCC features directly from an audio file.
    
    Args:
        file_path (str): Path to the audio file
        n_mfcc (int): Number of MFCCs to extract
        
    Returns:
        numpy.ndarray: MFCC features
    """
    try:
        audio_data, sampling_rate = load_audio(file_path)
        if audio_data is None:
            return None
        return extract_mfcc(audio_data, sampling_rate, n_mfcc=n_mfcc)
    except Exception as e:
        print(f"Error extracting MFCC features from file: {str(e)}")
        return None

def save_audio(file_path, audio_data, sampling_rate):
    """
    Save audio data to a file.
    
    Args:
        file_path (str): Path where to save the audio file
        audio_data (numpy.ndarray): Audio data
        sampling_rate (int): Sampling rate
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sf.write(file_path, audio_data, sampling_rate)
        return True
    except Exception as e:
        print(f"Error saving audio file {file_path}: {str(e)}")
        return False
        
def process_audio_file(input_file, output_file=None, apply_filter=True, sigma=2):
    """
    Process an audio file by applying filters if needed.
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str, optional): Path where to save the processed audio file.
            If None, the input file will be overwritten.
        apply_filter (bool): Whether to apply Gaussian filter
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        str: Path to the processed audio file
    """
    try:
        audio_data, sampling_rate = load_audio(input_file)
        
        if audio_data is None:
            return None
            
        if apply_filter:
            audio_data = apply_gaussian_filter(audio_data, sigma=sigma)
            
        save_path = output_file if output_file else input_file
        save_audio(save_path, audio_data, sampling_rate)
        
        return save_path
    except Exception as e:
        print(f"Error processing audio file {input_file}: {str(e)}")
        return None
