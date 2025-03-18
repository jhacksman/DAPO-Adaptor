# DAPO-Adaptor Verification Rollout

"""
This module implements the VerificationRollout class, which extends the vLLMRollout
class to add verification capabilities following the Sample, Scrutinize and Scale approach.
"""

from typing import List, Dict, Any, Optional, Union
import logging

# Import from verl framework
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from verl import DataProto
from tensordict import TensorDict

# Import from verification module
from .verification_utils import verify_response, compare_responses
from .prompts import VERIFICATION_PROMPT, COMPARISON_PROMPT

logger = logging.getLogger(__name__)


class VerificationRollout(vLLMRollout):
    """
    A rollout that implements the Sample, Scrutinize and Scale verification approach.
    
    This class extends the vLLMRollout class to add verification capabilities
    following Algorithm 1 from the SSaSEITSbSV paper.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the VerificationRollout.
        
        Args:
            *args: Arguments to pass to the parent class.
            **kwargs: Keyword arguments to pass to the parent class.
                verification_config: Configuration for the verification process.
        """
        # Extract verification config if provided
        self.verification_config = kwargs.pop('verification_config', {})
        
        # Default verification parameters
        self.kinf = self.verification_config.get('kinf', 200)
        self.kverif = self.verification_config.get('kverif', 50)
        self.ktie = self.verification_config.get('ktie', 100)
        self.threshold = self.verification_config.get('threshold', 0.05)
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        logger.info(f"Initialized VerificationRollout with kinf={self.kinf}, "
                   f"kverif={self.kverif}, ktie={self.ktie}, threshold={self.threshold}")
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences with verification.
        
        This method implements the three-stage verification process:
        1. Generate multiple candidate responses
        2. Verify each response
        3. Compare top candidates to break ties
        
        Args:
            prompts: The input prompts.
            
        Returns:
            The best response for each prompt.
        """
        logger.info(f"Generating sequences with verification for {len(prompts)} prompts")
        
        # Stage 1: Generate multiple candidate responses
        responses = self.generate_multiple_responses(prompts, self.kinf)
        logger.info(f"Generated {len(responses)} candidate responses")
        
        # Stage 2: Verify each response
        scores = self.verify_responses(responses, self.kverif)
        logger.info(f"Verified responses with scores: {scores}")
        
        # Find the top responses
        top_indices = self._find_top_responses(scores)
        logger.info(f"Found {len(top_indices)} top responses: {top_indices}")
        
        # Stage 3: Compare top candidates if there's a tie
        if len(top_indices) > 1:
            logger.info(f"Breaking tie between {len(top_indices)} responses")
            winner_idx = self.compare_responses([responses[i] for i in top_indices], self.ktie)
            best_response = responses[top_indices[winner_idx]]
        else:
            logger.info("No tie to break")
            best_response = responses[top_indices[0]]
        
        logger.info("Verification complete")
        return best_response
    
    def generate_multiple_responses(self, prompts: DataProto, kinf: int) -> List[DataProto]:
        """
        Generate multiple candidate responses for each prompt.
        
        Args:
            prompts: The input prompts.
            kinf: The number of responses to generate for each prompt.
            
        Returns:
            A list of DataProto objects, each containing a candidate response.
        """
        logger.info(f"Generating {kinf} responses for each prompt")
        
        # Configure sampling parameters for multiple response generation
        sampling_kwargs = {
            'n': kinf,  # Generate kinf responses per prompt
            'temperature': self.verification_config.get('temperature', 0.7),
            'top_p': self.verification_config.get('top_p', 0.9),
            'top_k': self.verification_config.get('top_k', 50),
            'do_sample': True
        }
        
        # Add do_sample flag to meta_info to ensure sampling is enabled
        prompts_with_sampling = DataProto(
            batch=prompts.batch,
            non_tensor_batch=prompts.non_tensor_batch,
            meta_info={**prompts.meta_info, 'do_sample': True}
        )
        
        # Use the parent class's generate_sequences method with modified sampling parameters
        with self.update_sampling_params(**sampling_kwargs):
            # Generate multiple responses using the vLLM engine
            all_responses = super().generate_sequences(prompts_with_sampling)
            
            # Extract individual responses from the batch
            batch_size = prompts.batch.batch_size[0]
            responses = []
            
            # The responses are interleaved in the output batch
            # We need to separate them into individual DataProto objects
            for i in range(kinf):
                # Extract responses for the current sample index
                indices = list(range(i, batch_size * kinf, kinf))
                
                # Create a new DataProto for each response
                response_batch = TensorDict({
                    key: all_responses.batch[key][indices] 
                    for key in all_responses.batch.keys()
                }, batch_size=(batch_size,))
                
                # Create non_tensor_batch if needed
                response_non_tensor = {}
                for key, val in all_responses.non_tensor_batch.items():
                    response_non_tensor[key] = val[indices]
                
                # Create a DataProto for the current response
                response = DataProto(
                    batch=response_batch,
                    non_tensor_batch=response_non_tensor,
                    meta_info=all_responses.meta_info
                )
                
                responses.append(response)
        
        logger.info(f"Generated {len(responses)} responses")
        return responses
    
    def verify_responses(self, responses: List[DataProto], kverif: int) -> List[float]:
        """
        Verify the correctness of each candidate response.
        
        Args:
            responses: The candidate responses to verify.
            kverif: The number of verification samples to generate for each response.
            
        Returns:
            A list of verification scores for each response.
        """
        logger.info(f"Verifying {len(responses)} responses with {kverif} samples each")
        
        # Use the verification_utils.verify_response function to generate 
        # verification scores for each response
        scores = []
        
        # Get the appropriate verification prompt template based on the task type
        prompt_template = self.verification_config.get('verification_prompt', None)
        
        # Process responses in batches for efficiency
        batch_size = self.verification_config.get('verification_batch_size', 10)
        
        for i in range(0, len(responses), batch_size):
            batch_responses = responses[i:i+batch_size]
            batch_scores = []
            
            # Process each response in the current batch
            for response in batch_responses:
                # Use the verify_response function from verification_utils
                # Pass the inference engine as the model for verification
                score = verify_response(
                    response=response,
                    model=self.inference_engine,
                    kverif=kverif,
                    prompt_template=prompt_template
                )
                batch_scores.append(score)
            
            # Add batch scores to the overall scores list
            scores.extend(batch_scores)
            
            logger.info(f"Verified batch {i//batch_size + 1}/{(len(responses) + batch_size - 1)//batch_size} "
                       f"with scores: {batch_scores}")
        
        # Normalize scores if needed
        if self.verification_config.get('normalize_scores', False):
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [(score - min_score) / (max_score - min_score) for score in scores]
                    logger.info(f"Normalized scores to range [0, 1]")
        
        logger.info(f"Verification complete with scores: {scores}")
        return scores
    
    def compare_responses(self, top_responses: List[DataProto], ktie: int) -> int:
        """
        Compare the top candidate responses to break ties.
        
        This method implements a tournament-style comparison between the top responses
        to determine the best one. It follows Algorithm 1 from the SSaSEITSbSV paper,
        specifically the tie-breaking stage.
        
        Args:
            top_responses: The top candidate responses to compare.
            ktie: The number of comparison samples to generate for each pair.
            
        Returns:
            The index of the winning response.
        """
        logger.info(f"Comparing {len(top_responses)} top responses with {ktie} samples each")
        
        # If there's only one response, return it
        if len(top_responses) == 1:
            return 0
        
        # Get the comparison prompt template if specified
        prompt_template = self.verification_config.get('comparison_prompt', None)
        
        # Initialize win counts for each response
        win_counts = [0] * len(top_responses)
        
        # Perform pairwise comparisons between all top responses
        for i in range(len(top_responses)):
            for j in range(i + 1, len(top_responses)):
                logger.info(f"Comparing response {i} with response {j}")
                
                # Use the compare_responses function from verification_utils
                # to determine the winner of this pair
                winner = compare_responses(
                    response1=top_responses[i],
                    response2=top_responses[j],
                    model=self.inference_engine,
                    ktie=ktie,
                    prompt_template=prompt_template
                )
                
                # Increment the win count for the winner
                if winner == 0:
                    win_counts[i] += 1
                    logger.info(f"Response {i} won against response {j}")
                else:
                    win_counts[j] += 1
                    logger.info(f"Response {j} won against response {i}")
        
        # Find the response with the most wins
        max_wins = max(win_counts)
        winners = [i for i, wins in enumerate(win_counts) if wins == max_wins]
        
        # If there's still a tie, choose the response with the highest verification score
        if len(winners) > 1:
            logger.info(f"Multiple winners with {max_wins} wins each: {winners}")
            
            # In case of a tie, return the first winner
            # This is a simplification; in a more sophisticated implementation,
            # we could use additional criteria to break the tie
            winner_idx = winners[0]
        else:
            winner_idx = winners[0]
        
        logger.info(f"Comparison complete with win counts: {win_counts} and winner index: {winner_idx}")
        return winner_idx
    
    def _find_top_responses(self, scores: List[float]) -> List[int]:
        """
        Find the indices of the top responses based on their verification scores.
        
        Args:
            scores: The verification scores for each response.
            
        Returns:
            A list of indices of the top responses.
        """
        if not scores:
            return [0]
        
        # Find the maximum score
        max_score = max(scores)
        
        # Find all responses with scores within the threshold of the maximum
        top_indices = [i for i, score in enumerate(scores) if score >= max_score - self.threshold]
        
        logger.info(f"Found {len(top_indices)} top responses with scores within {self.threshold} of {max_score}")
        return top_indices
