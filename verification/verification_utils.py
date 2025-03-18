# DAPO-Adaptor Verification Utilities

"""
This module provides utility functions for the verification process,
including functions for verifying responses and comparing responses.
"""

from typing import List, Dict, Any, Optional, Union
import logging

# For testing purposes, create a mock DataProto class if verl is not available
try:
    from verl import DataProto
except ImportError:
    # Mock class for testing
    class DataProto:
        def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
            self.batch = batch or {}
            self.non_tensor_batch = non_tensor_batch or {}
            self.meta_info = meta_info or {}
from tensordict import TensorDict

# Import from verification module
from .prompts import (
    VERIFICATION_PROMPT, COMPARISON_PROMPT,
    MATH_VERIFICATION_PROMPT, LOGIC_VERIFICATION_PROMPT, FACTUAL_VERIFICATION_PROMPT
)

logger = logging.getLogger(__name__)


def verify_response(response: DataProto, model, kverif: int, prompt_template: str = None) -> float:
    """
    Verify a single response and return a verification score.
    
    This function uses the model to generate verification scores for a response.
    It generates kverif samples and returns the average score.
    
    Args:
        response: The response to verify.
        model: The model to use for verification.
        kverif: The number of verification samples to generate.
        prompt_template: The prompt template to use for verification. If None,
                         the default VERIFICATION_PROMPT will be used.
        
    Returns:
        The average verification score (between 0 and 1).
    """
    logger.info(f"Verifying response with {kverif} samples")
    
    # Extract question and response text from DataProto
    question = response.non_tensor_batch.get('question', '')
    if not question:
        # Try to extract from meta_info if not in non_tensor_batch
        question = response.meta_info.get('question', '')
    
    # Extract the response text from the response tensor
    # Check for different possible field names based on VERL's data structure
    response_text = None
    for field in ['responses', 'response', 'output', 'generated_text']:
        if field in response.batch:
            response_text = response.batch[field]
            break
    
    if response_text is None:
        logger.warning("No response text found in DataProto batch fields")
        # Try non_tensor_batch as a fallback
        for field in ['responses', 'response', 'output', 'generated_text']:
            if field in response.non_tensor_batch:
                response_text = response.non_tensor_batch[field]
                break
    
    if response_text is None:
        logger.warning("No response text found in DataProto")
        return 0.0
    
    # Use the appropriate prompt template based on the task type
    if prompt_template is None:
        # Determine the appropriate verification prompt based on the question type
        if "math" in question.lower() or any(op in question for op in ['+', '-', '*', '/', '=', '<', '>']):
            prompt_template = MATH_VERIFICATION_PROMPT
        elif "logic" in question.lower() or any(term in question.lower() for term in ["if", "then", "therefore", "because"]):
            prompt_template = LOGIC_VERIFICATION_PROMPT
        elif "fact" in question.lower() or any(term in question.lower() for term in ["who", "what", "when", "where"]):
            prompt_template = FACTUAL_VERIFICATION_PROMPT
        else:
            prompt_template = VERIFICATION_PROMPT
    
    # Format the verification prompt
    verification_prompt = prompt_template.format(
        question=question,
        response=response_text
    )
    
    # Create a DataProto for the verification prompt
    verification_data = DataProto(
        batch={},
        non_tensor_batch={'prompt': verification_prompt},
        meta_info={'do_sample': True}
    )
    
    # Generate verification samples
    verification_scores = []
    for i in range(kverif):
        # Generate a verification sample
        verification_result = model.generate_sequences(verification_data)
        
        # Extract the verification score (0 or 1) from the result
        # The model should return 1 for correct and 0 for incorrect
        result_text = None
        for field in ['responses', 'response', 'output', 'generated_text']:
            if field in verification_result.batch:
                result_text = verification_result.batch[field]
                break
            
        if result_text is None:
            # Try non_tensor_batch as a fallback
            for field in ['responses', 'response', 'output', 'generated_text']:
                if field in verification_result.non_tensor_batch:
                    result_text = verification_result.non_tensor_batch[field]
                    break
        
        if result_text is None:
            logger.warning("Could not extract verification result")
            continue
            
        result_text = str(result_text).strip()
        
        # Parse the verification result
        try:
            # Extract the first digit (0 or 1) from the result
            for char in result_text:
                if char.isdigit():
                    score = int(char)
                    if score in [0, 1]:
                        verification_scores.append(score)
                        break
            else:
                # If no digit found, check for keywords
                if any(term in result_text.lower() for term in ["correct", "yes", "true"]):
                    verification_scores.append(1)
                elif any(term in result_text.lower() for term in ["incorrect", "no", "false"]):
                    verification_scores.append(0)
                else:
                    logger.warning(f"Could not parse verification result: {result_text}")
                    # Default to 0 if parsing fails
                    verification_scores.append(0)
        except Exception as e:
            logger.error(f"Error parsing verification result: {e}")
            verification_scores.append(0)
    
    # Calculate the average verification score
    if verification_scores:
        average_score = sum(verification_scores) / len(verification_scores)
    else:
        average_score = 0.0
    
    logger.info(f"Verification complete with {len(verification_scores)} valid samples and average score: {average_score}")
    return average_score


def compare_responses(response1: DataProto, response2: DataProto, model, ktie: int, prompt_template: str = None) -> int:
    """
    Compare two responses and return the index of the winner (0 or 1).
    
    This function uses the model to compare two responses and determine
    which one is better. It generates ktie samples and returns the index
    of the response that wins the most comparisons.
    
    Args:
        response1: The first response to compare.
        response2: The second response to compare.
        model: The model to use for comparison.
        ktie: The number of comparison samples to generate.
        prompt_template: The prompt template to use for comparison. If None,
                         the default COMPARISON_PROMPT will be used.
        
    Returns:
        The index of the winning response (0 for response1, 1 for response2).
    """
    logger.info(f"Comparing two responses with {ktie} samples")
    
    # Extract question from the responses
    question = response1.non_tensor_batch.get('question', '')
    if not question:
        # Try to extract from meta_info if not in non_tensor_batch
        question = response1.meta_info.get('question', '')
    
    # Extract the response texts from the response tensors
    response1_text = None
    response2_text = None
    
    # Extract response1 text
    for field in ['responses', 'response', 'output', 'generated_text']:
        if field in response1.batch:
            response1_text = response1.batch[field]
            break
    
    if response1_text is None:
        # Try non_tensor_batch as a fallback
        for field in ['responses', 'response', 'output', 'generated_text']:
            if field in response1.non_tensor_batch:
                response1_text = response1.non_tensor_batch[field]
                break
    
    # Extract response2 text
    for field in ['responses', 'response', 'output', 'generated_text']:
        if field in response2.batch:
            response2_text = response2.batch[field]
            break
    
    if response2_text is None:
        # Try non_tensor_batch as a fallback
        for field in ['responses', 'response', 'output', 'generated_text']:
            if field in response2.non_tensor_batch:
                response2_text = response2.non_tensor_batch[field]
                break
    
    if response1_text is None or response2_text is None:
        logger.warning("One or both response texts not found in DataProto")
        return 0  # Default to first response if we can't compare
    
    # Use the provided prompt template or the default one
    if prompt_template is None:
        prompt_template = COMPARISON_PROMPT
    
    # Format the comparison prompt
    comparison_prompt = prompt_template.format(
        question=question,
        response1=response1_text,
        response2=response2_text
    )
    
    # Create a DataProto for the comparison prompt
    comparison_data = DataProto(
        batch={},
        non_tensor_batch={'prompt': comparison_prompt},
        meta_info={'do_sample': True}
    )
    
    # Generate comparison samples
    votes = [0, 0]  # Votes for response1 and response2
    
    for i in range(ktie):
        # Generate a comparison sample
        comparison_result = model.generate_sequences(comparison_data)
        
        # Extract the comparison result from the output
        result_text = None
        for field in ['responses', 'response', 'output', 'generated_text']:
            if field in comparison_result.batch:
                result_text = comparison_result.batch[field]
                break
        
        if result_text is None:
            # Try non_tensor_batch as a fallback
            for field in ['responses', 'response', 'output', 'generated_text']:
                if field in comparison_result.non_tensor_batch:
                    result_text = comparison_result.non_tensor_batch[field]
                    break
        
        if result_text is None:
            logger.warning("Could not extract comparison result")
            continue
            
        result_text = str(result_text).strip()
        
        # Parse the comparison result
        try:
            # Look for "1" or "2" in the result
            if "1" in result_text and "2" not in result_text:
                votes[0] += 1
            elif "2" in result_text and "1" not in result_text:
                votes[1] += 1
            else:
                # Check for keywords indicating preference
                if any(term in result_text.lower() for term in ["first", "response 1", "response1"]):
                    votes[0] += 1
                elif any(term in result_text.lower() for term in ["second", "response 2", "response2"]):
                    votes[1] += 1
                else:
                    logger.warning(f"Could not parse comparison result: {result_text}")
                    # Split the vote evenly in case of ambiguity
                    votes[0] += 0.5
                    votes[1] += 0.5
        except Exception as e:
            logger.error(f"Error parsing comparison result: {e}")
            # Split the vote evenly in case of error
            votes[0] += 0.5
            votes[1] += 0.5
    
    # Determine the winner based on votes
    winner_idx = 0 if votes[0] >= votes[1] else 1
    
    logger.info(f"Comparison complete with votes: {votes} and winner index: {winner_idx}")
    return winner_idx


def calculate_self_consistency(responses: List[str]) -> float:
    """
    Calculate the self-consistency of responses.
    
    This function calculates the percentage of responses that are consistent
    with the most common response.
    
    Args:
        responses: A list of response strings.
        
    Returns:
        The self-consistency score.
    """
    logger.info(f"Calculating self-consistency for {len(responses)} responses")
    
    if not responses:
        logger.warning("No responses provided for self-consistency calculation")
        return 0.0
    
    if len(responses) == 1:
        logger.info("Only one response provided, self-consistency is 1.0")
        return 1.0
    
    # Count occurrences of each response
    response_counts = {}
    for response in responses:
        # Normalize text for comparison
        normalized_response = str(response).strip().lower()
        response_counts[normalized_response] = response_counts.get(normalized_response, 0) + 1
    
    # Find the most common response
    most_common_response = max(response_counts.items(), key=lambda x: x[1])
    most_common_count = most_common_response[1]
    
    # Calculate consistency score
    consistency = most_common_count / len(responses)
    
    logger.info(f"Self-consistency calculation complete with score: {consistency}")
    return consistency


def calculate_logical_consistency(response: str, constraints: Dict[str, List[str]] = None) -> float:
    """
    Calculate the logical consistency of a response.
    
    This function checks if a response is logically consistent based on
    a set of constraints such as required keywords, forbidden words, and patterns.
    
    Args:
        response: The response text to check for logical consistency.
        constraints: A dictionary of constraints to check against.
            If None, a default set of constraints will be used.
        
    Returns:
        The logical consistency score (between 0 and 1).
    """
    logger.info(f"Calculating logical consistency for response")
    
    if not response:
        logger.warning("No response provided for logical consistency calculation")
        return 0.0
    
    if constraints is None:
        # Default constraints
        constraints = {
            'keywords': ['because', 'therefore', 'since', 'reason'],
            'forbidden': ['contradiction', 'impossible', 'cannot be both'],
            'patterns': [r'\d+', r'[A-Za-z]+ is [A-Za-z]+'],
        }
    
    # Normalize text for analysis
    response_text = str(response).strip().lower()
    
    # Calculate scores for each constraint type
    scores = []
    
    # Check for required keywords
    if 'keywords' in constraints and constraints['keywords']:
        keyword_matches = sum(1 for keyword in constraints['keywords'] if keyword.lower() in response_text)
        keyword_score = keyword_matches / len(constraints['keywords'])
        scores.append(keyword_score)
    
    # Check for forbidden words
    if 'forbidden' in constraints and constraints['forbidden']:
        forbidden_matches = sum(1 for word in constraints['forbidden'] if word.lower() in response_text)
        forbidden_score = 1.0 - (forbidden_matches / len(constraints['forbidden']))
        scores.append(forbidden_score)
    
    # Check for required patterns
    if 'patterns' in constraints and constraints['patterns']:
        import re
        pattern_matches = sum(1 for pattern in constraints['patterns'] if re.search(pattern, response_text))
        pattern_score = pattern_matches / len(constraints['patterns'])
        scores.append(pattern_score)
    
    # Calculate overall consistency score
    if scores:
        consistency = sum(scores) / len(scores)
    else:
        consistency = 0.0
    
    logger.info(f"Logical consistency calculation complete with score: {consistency}")
    return consistency
