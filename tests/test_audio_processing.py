"""
Unit tests for audio processing module
"""
import os
import sys
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processing import apply_gaussian_filter, extract_mfcc


class TestAudioProcessing(unittest.TestCase):
    """Test cases for audio processing functionality"""
    
    def test_gaussian_filter(self):
        """Test that Gaussian filter maintains array dimensions"""
        # Create test data
        test_data = np.ones(100)
        
        # Apply filter
        filtered_data = apply_gaussian_filter(test_data)
        
        # Check output has same dimensions as input
        self.assertEqual(filtered_data.shape, test_data.shape)
        
        # Check that filter actually modified the data (sum should be the same, but values different)
        self.assertAlmostEqual(np.sum(filtered_data), np.sum(test_data), places=5)
        self.assertFalse(np.array_equal(filtered_data, test_data))
    
    def test_gaussian_filter_2d_input(self):
        """Test that Gaussian filter works with 2D input"""
        # Create 2D test data
        test_data_2d = np.ones((10, 10))
        
        # This should raise an error since our implementation expects 1D audio
        with self.assertRaises(Exception):
            apply_gaussian_filter(test_data_2d)
    
    def test_extract_mfcc_invalid_input(self):
        """Test MFCC extraction with invalid input"""
        # None input should return None
        self.assertIsNone(extract_mfcc(None, 22050))
        
        # Empty array should raise an error or return None
        with self.assertRaises(Exception):
            extract_mfcc(np.array([]), 22050)


if __name__ == '__main__':
    unittest.main()
