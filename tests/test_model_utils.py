"""
Unit tests for model utilities module
"""
import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_utils import get_all_emotions, find_model_file


class TestModelUtils(unittest.TestCase):
    """Test cases for model utilities"""
    
    def test_get_all_emotions(self):
        """Test that emotion list is returned correctly"""
        emotions = get_all_emotions()
        
        # Check that emotions are returned as a list
        self.assertIsInstance(emotions, list)
        
        # Check that all expected emotions are present
        expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
        for emotion in expected_emotions:
            self.assertIn(emotion, emotions)
    
    def test_find_model_file_nonexistent(self):
        """Test finding a model file that doesn't exist"""
        # Set search directories to nonexistent paths
        search_dirs = ['/nonexistent/path', '/another/nonexistent/path']
        
        # Should return None when model not found
        result = find_model_file(model_name='nonexistent_model.keras', search_dirs=search_dirs)
        self.assertIsNone(result)
    
    @patch('os.path.exists')
    def test_find_model_file_exists(self, mock_exists):
        """Test finding a model file that exists"""
        # Mock os.path.exists to return True for one path
        mock_exists.side_effect = lambda path: 'fake/path/models/my_model.keras' in path
        
        # Set search directories
        search_dirs = ['/nonexistent/path', 'fake/path/models']
        
        # Should find the model
        result = find_model_file(model_name='my_model.keras', search_dirs=search_dirs)
        self.assertEqual(result, 'fake/path/models/my_model.keras')


if __name__ == '__main__':
    unittest.main()
