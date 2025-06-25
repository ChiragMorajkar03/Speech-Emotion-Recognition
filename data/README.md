# Data Files

This directory is intended to store audio data files used for training, testing, or demonstration purposes.

## Datasets

The Speech Emotion Recognition model was trained using the following datasets:

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - Contains speech recordings from 24 professional actors (12 female, 12 male)
   - Emotions: calm, happy, sad, angry, fearful, surprise, and disgust
   - [Download RAVDESS](https://zenodo.org/record/1188976)

2. **TESS** (Toronto Emotional Speech Set)
   - Contains speech recordings from 2 actresses
   - Age ranges: young adult and elderly
   - Emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral
   - [Download TESS](https://tspace.library.utoronto.ca/handle/1807/24487)

3. **SAVEE** (Surrey Audio-Visual Expressed Emotion)
   - Contains speech recordings from 4 male actors
   - Emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral
   - [Download SAVEE](http://personal.ee.surrey.ac.uk/Personal/P.Jackson/SAVEE/Download.html)

## Data Format

The audio files should be in WAV format for best compatibility with the application.

## Excluded from Repository

Due to size constraints and copyright considerations, the audio data files are not included in this repository.

## Sample Audio

For demonstration purposes, we recommend creating a small collection of sample audio files that showcase different emotions:

- Create a `samples` subdirectory in this folder
- Include 1-2 examples for each emotion (angry, disgust, fear, happy, neutral, surprise, sad)
- Keep files short (2-3 seconds each)
- Use files that you have permission to distribute

## Using Your Own Data

You can use your own audio files with the application:
1. Ensure they are in WAV, MP3, or OGG format
2. Files should contain clear speech with minimal background noise
3. Short audio clips work best (2-5 seconds)
