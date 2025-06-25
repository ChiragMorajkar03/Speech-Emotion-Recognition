# Contributing to Speech Emotion Recognition

Thank you for your interest in contributing to the Speech Emotion Recognition project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your forked repository locally
3. Install dependencies: `pip install -r requirements.txt`
4. Create a new branch for your feature or bug fix

## Project Structure

```
speech-emotion-recognition/
├── app.py                           # Main Streamlit application
├── src/                             # Source code modules
│   ├── __init__.py                  # Package initialization
│   ├── audio_processing.py          # Audio processing utilities
│   └── model_utils.py               # Model loading and prediction utilities
├── models/                          # Directory for model files
│   └── README.md                    # Instructions for model files
├── data/                            # Data files (not included in repo)
│   └── README.md                    # Instructions for obtaining data
├── examples/                        # Example files
├── .github/                         # GitHub configuration
│   └── workflows/                   # GitHub Actions workflows
├── requirements.txt                 # Project dependencies
├── setup.py                         # Package installation configuration
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines for Python code
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

### Pull Requests

1. Create a descriptive pull request title
2. Reference any related issues
3. Explain the changes you've made and why
4. Make sure all tests pass
5. Keep PRs focused on a single concern

### Testing

Before submitting a PR:
1. Run tests: `pytest`
2. Verify the app works: `streamlit run app.py`
3. Check code style: `flake8`

## Working with Large Files

This project uses large model files that are not directly included in the repository. See the instructions in [models/README.md](models/README.md) for details on obtaining these files.

## Code of Conduct

- Be respectful and inclusive
- Give constructive feedback
- Focus on the best interests of the project

## Questions?

If you have any questions, feel free to open an issue or contact the maintainers.

Thank you for contributing!
