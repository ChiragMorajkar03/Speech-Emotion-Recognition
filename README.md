# Speech Emotion Recognition

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.8+-red.svg)](https://streamlit.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning application that recognizes human emotions from speech using audio signal processing and deep learning techniques.

## Overview
This project implements a speech emotion recognition system that can identify emotions such as anger, happiness, sadness, fear, disgust, and neutral states from audio inputs. It uses Mel-frequency cepstral coefficients (MFCCs) for feature extraction and a neural network model for classification.

## Features
- Live audio recording for real-time emotion detection
- Upload audio files for emotion analysis
- Audio preprocessing with Gaussian filtering
- Interactive web interface using Streamlit
- Pre-trained deep learning model for emotion classification

## Demo
![Speech Emotion Recognition Demo](https://github.com/yourusername/speech-emotion-recognition/raw/main/demo.gif)

## Model Architecture
The emotion recognition model is built using TensorFlow with the following architecture:
- Multiple convolutional layers for feature extraction
- Dropout layers to prevent overfitting
- Dense layers for classification
- Trained on a dataset of labeled emotional speech samples

## Technologies Used
- TensorFlow for deep learning model
- Librosa for audio processing and feature extraction
- Streamlit for web interface
- SoundFile and SoundDevice for audio handling
- Numpy for numerical operations
- Scipy for signal processing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/speech-emotion-recognition.git

# Navigate to project directory
cd speech-emotion-recognition

# Install required packages
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app2.py
```

The app provides two methods for emotion recognition:
1. **Upload Audio**: Upload a WAV, MP3, or OGG file for analysis
2. **Record Audio**: Record a 3-second audio clip directly in the app

## Dataset
The model was trained on standard emotion recognition datasets including:
- Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- Toronto Emotional Speech Set (TESS)
- Surrey Audio-Visual Expressed Emotion (SAVEE)

## Model Performance
- Overall accuracy: ~75% on test data
- Best recognized emotions: Angry, Happy, Neutral
- Most challenging emotions: Fear, Disgust

## Project Structure
```
speech-emotion-recognition/
├── app2.py                         # Streamlit application
├── Speech Emotion Recognition.ipynb # Model development notebook
├── my_model.json                   # Model architecture
├── my_model.keras                  # Trained model
├── my_model_weights.weights.h5     # Model weights
└── requirements.txt                # Dependencies
```

## Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_audio_processing
```

Test coverage reports can be generated with:

```bash
coverage run -m unittest discover
coverage report
```

## Code Quality

This project follows best practices for Python code quality:

- **PEP 8 compliant** code style throughout
- **Type hints** for improved IDE integration and code safety
- **Comprehensive docstrings** following Google style
- **Modular architecture** with separation of concerns
- **Error handling** with informative user feedback

## Roadmap

### Short-term Goals
- [ ] Real-time continuous emotion monitoring
- [ ] Improved accuracy for challenging emotions (fear, disgust)
- [ ] Pre-trained models for different languages

### Medium-term Goals
- [ ] Web API deployment with FastAPI
- [ ] Mobile application integration
- [ ] Batch processing for audio files

### Long-term Vision
- [ ] Multimodal emotion recognition (combining speech and facial expressions)
- [ ] Sentiment trend analysis over time
- [ ] Emotion-based recommendation systems integration

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The academic papers and research that established the foundation for speech emotion recognition
- Open-source emotional speech datasets used for training
- The Streamlit, TensorFlow, and Librosa development communities
