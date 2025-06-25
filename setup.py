from setuptools import setup, find_packages

setup(
    name="speech_emotion_recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.22.0",
        "librosa>=0.9.1",
        "streamlit>=1.8.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "pandas>=1.4.1",
        "soundfile>=0.10.3",
        "scipy>=1.8.0",
        "sounddevice>=0.4.4",
    ],
    author="Chirag Morajkar",
    author_email="your.email@example.com",
    description="A speech emotion recognition system using deep learning",
    keywords="speech, emotion, recognition, deep learning, audio processing",
    url="https://github.com/yourusername/speech-emotion-recognition",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
