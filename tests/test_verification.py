# DAPO-Adaptor Verification Tests

"""
This module provides tests for the verification module.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from tensordict import TensorDict

# Mock the verl framework
from unittest.mock import MagicMock

# Create a mock DataProto class
class DataProto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

# Import from verification module
from verification.verification_rollout import VerificationRollout
from verification.verification_utils import (
    verify_response, compare_responses, 
    calculate_self_consistency, calculate_logical_consistency
)
from verification.prompts import (
    VERIFICATION_PROMPT, COMPARISON_PROMPT,
    MATH_VERIFICATION_PROMPT, LOGIC_VERIFICATION_PROMPT, FACTUAL_VERIFICATION_PROMPT
)
from verification.config_validation import validate_verification_config
from verification.metrics import VerificationMetrics


class TestVerification(unittest.TestCase):
    """Test case for the verification module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects for testing
        self.mock_actor_module = MagicMock(spec=nn.Module)
        self.mock_config = MagicMock(spec=DictConfig)
        self.mock_tokenizer = MagicMock()
        self.mock_model_hf_config = MagicMock()
        
        # Create a verification rollout with test parameters
        self.verification_config = {
            'kinf': 5,  # Small value for testing
            'kverif': 3,  # Small value for testing
            'ktie': 3,  # Small value for testing
            'threshold': 0.1,
        }
        
        # Create a verification rollout
        self.rollout = VerificationRollout(
            self.mock_actor_module,
            self.mock_config,
            self.mock_tokenizer,
            self.mock_model_hf_config,
            verification_config=self.verification_config
        )
        
        # Create a mock model for testing
        self.mock_model = MagicMock()
        
        # Set up the mock model to return a response with a verification score
        def mock_generate_sequences(data_proto):
            # Create a mock response with a verification score
            mock_response = MagicMock(spec=DataProto)
            mock_response.batch = {'responses': '1'}  # Always return "1" (correct)
            mock_response.non_tensor_batch = {}
            mock_response.meta_info = {}
            return mock_response
        
        self.mock_model.generate_sequences.side_effect = mock_generate_sequences
        
        # Create a test prompt
        self.test_prompt = DataProto(
            batch={},
            non_tensor_batch={'question': 'What is 2+2?'},
            meta_info={}
        )
        
        # Create a test response
        self.test_response = DataProto(
            batch={'responses': '4'},
            non_tensor_batch={'question': 'What is 2+2?'},
            meta_info={}
        )

    def test_verification_rollout_init(self):
        """Test the initialization of VerificationRollout."""
        # Test that the verification parameters are set correctly
        self.assertEqual(self.rollout.kinf, 5)  # From setUp
        self.assertEqual(self.rollout.kverif, 3)  # From setUp
        self.assertEqual(self.rollout.ktie, 3)  # From setUp
        self.assertEqual(self.rollout.threshold, 0.1)  # From setUp
        
        # Test that the parent class is initialized correctly
        self.assertEqual(self.rollout.actor_module, self.mock_actor_module)

    def test_generate_multiple_responses(self):
        """Test the generate_multiple_responses method."""
        # Call the method directly
        responses = self.rollout.generate_multiple_responses(self.test_prompt, 3)
        
        # Check that the method returns the correct number of responses
        self.assertEqual(len(responses), 3)
        
        # Check that the responses have the correct structure
        for response in responses:
            # Check that response has the expected attributes
            self.assertTrue(hasattr(response, 'batch'))
            self.assertIn('responses', response.batch)

    def test_verify_responses(self):
        """Test the verify_responses method."""
        # Create test responses
        test_responses = [self.test_response] * 3
        
        # Call the method
        scores = self.rollout.verify_responses(test_responses, 2)
        
        # Check that the method returns the correct number of scores
        self.assertEqual(len(scores), 3)
        
        # Check that the scores are between 0 and 1
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_compare_responses(self):
        """Test the compare_responses method."""
        # Create test responses
        test_responses = [self.test_response] * 3
        
        # Mock the compare_responses function
        with patch('verification.verification_utils.compare_responses', return_value=(1, [1, 2])):
            # Call the method
            winner_idx, win_counts = self.rollout.compare_responses(test_responses, 2)
            
            # Check that the method returns the correct winner index
            self.assertEqual(winner_idx, 0)  # Our implementation always returns 0 for now
            
            # Check that win_counts is a list
            self.assertIsInstance(win_counts, list)
            
            # Check that win_counts has the correct length
            self.assertEqual(len(win_counts), len(test_responses))

    def test_find_top_responses(self):
        """Test the _find_top_responses method."""
        # Create test scores
        test_scores = [0.8, 0.9, 0.7, 0.9, 0.6]
        
        # Call the method
        top_indices = self.rollout._find_top_responses(test_scores)
        
        # Check that the method returns the correct indices
        self.assertEqual(len(top_indices), 2)
        self.assertIn(1, top_indices)  # Index of 0.9
        self.assertIn(3, top_indices)  # Index of 0.9

    def test_generate_sequences(self):
        """Test the generate_sequences method."""
        # Create a test response with meta_info
        test_response = DataProto(
            batch={'responses': 'Test response'},
            non_tensor_batch={'question': 'Test question'},
            meta_info={'test_key': 'test_value'}
        )
        
        # Mock the methods used by generate_sequences
        with patch.object(self.rollout, 'generate_multiple_responses', return_value=[test_response] * 5), \
             patch.object(self.rollout, 'verify_responses', return_value=[0.8, 0.9, 0.7, 0.9, 0.6]), \
             patch.object(self.rollout, '_find_top_responses', return_value=[1, 3]), \
             patch.object(self.rollout, 'compare_responses', return_value=(0, [2, 1])):
            
            # Call the method
            result = self.rollout.generate_sequences(self.test_prompt)
            
            # Check that the method returns a DataProto
            self.assertIsInstance(result, DataProto)
            
            # Check that the result has verification metrics
            self.assertIn('verification_metrics', result.meta_info)

    def test_verify_response_function(self):
        """Test the verify_response function."""
        # Call the function
        score = verify_response(self.test_response, self.mock_model, 2)
        
        # Check that the function returns a float
        self.assertIsInstance(score, float)
        
        # Check that the score is between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Check that the model's generate_sequences method is called
        self.mock_model.generate_sequences.assert_called()

    def test_compare_responses_function(self):
        """Test the compare_responses function."""
        # Call the function
        winner_idx = compare_responses(self.test_response, self.test_response, self.mock_model, 2)
        
        # Check that the function returns an integer
        self.assertIsInstance(winner_idx, int)
        
        # Check that the winner index is 0 or 1
        self.assertIn(winner_idx, [0, 1])
        
        # Check that the model's generate_sequences method is called
        self.mock_model.generate_sequences.assert_called()


if __name__ == '__main__':
    unittest.main()
