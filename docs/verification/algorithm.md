# Algorithm Implementation

This document provides a detailed explanation of the algorithm implementation for the Sample, Scrutinize and Scale verification approach.

## Algorithm 1: Sampling-Based Search with Verification

The core algorithm from the SSaSEITSbSV paper is as follows:

```
Algorithm 1 Sampling-Based Search (Verification@kinf)
Require: Prompt Q, language model LM, scaling parameters kinf, kverif, ktie.
1: Populate S with kinf samples from LM("Answer Q").
   ▷ Stage 1: Generate Responses
2: for each candidate response si ∈ S do
   ▷ Stage 2: Verify Responses
3:   Populate Vi with kverif samples from LM("Return 1[response si to Q is correct]").
4: Gather the highest-scored response SBest = {si | i ∈ [kinf], Avg(Vi) ≥ maxj∈[kinf] Avg(Vj) − 0.05}.
5: if |SBest| = 1 then
6:   Return response si* where i* = maxj∈[kinf] Avg(Vj).
7: else
8:   for each pair of candidate responses (si, sj) ∈ SBest do
     ▷ Tie-Break: Compare Responses
9:     Populate Ci,j with ktie samples from LM("Which of responses {si, sj} to Q is correct?").
10:  Return response si* where i* is the winner of the most matchups {Ci,j | si, sj ∈ SBest}.
```

## Implementation Details

### Stage 1: Generate Responses

The first stage involves generating multiple candidate responses for a given prompt. This will be implemented by extending the `generate_sequences` method in the `vLLMRollout` class to generate `kinf` samples for each prompt.

```python
def generate_multiple_responses(self, prompts: DataProto, kinf: int) -> List[DataProto]:
    """Generate multiple candidate responses for each prompt.
    
    Args:
        prompts: The input prompts
        kinf: The number of responses to generate for each prompt
        
    Returns:
        A list of DataProto objects, each containing a candidate response
    """
    responses = []
    with self.update_sampling_params(n=kinf):
        output = self.inference_engine.generate(
            prompts=None,
            sampling_params=self.sampling_params,
            prompt_token_ids=idx_list,
            use_tqdm=False)
        # Process output to create a list of DataProto objects
        # ...
    return responses
```

### Stage 2: Verify Responses

The second stage involves verifying each candidate response. This will be implemented as a new method that uses the model to generate verification scores for each response.

```python
def verify_responses(self, responses: List[DataProto], kverif: int) -> List[float]:
    """Verify the correctness of each candidate response.
    
    Args:
        responses: The candidate responses to verify
        kverif: The number of verification samples to generate for each response
        
    Returns:
        A list of verification scores for each response
    """
    scores = []
    for response in responses:
        # Generate kverif verification samples for the response
        # Compute the average verification score
        # ...
        scores.append(avg_score)
    return scores
```

### Stage 3: Compare Top Candidates

The third stage involves comparing the top candidates to break ties. This will be implemented as a new method that performs pairwise comparisons between the top candidates.

```python
def compare_responses(self, top_responses: List[DataProto], ktie: int) -> int:
    """Compare the top candidate responses to break ties.
    
    Args:
        top_responses: The top candidate responses to compare
        ktie: The number of comparison samples to generate for each pair
        
    Returns:
        The index of the winning response
    """
    # Perform pairwise comparisons between all top responses
    # Count the number of wins for each response
    # Return the index of the response with the most wins
    # ...
```

## Integration with DAPO

The algorithm will be integrated with the DAPO framework by creating a new `VerificationRollout` class that extends the `vLLMRollout` class and implements the three-stage verification process.

```python
class VerificationRollout(vLLMRollout):
    """A rollout that implements the Sample, Scrutinize and Scale verification approach."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize verification parameters
        
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences with verification.
        
        This method implements the three-stage verification process:
        1. Generate multiple candidate responses
        2. Verify each response
        3. Compare top candidates to break ties
        """
        # Stage 1: Generate multiple candidate responses
        responses = self.generate_multiple_responses(prompts, kinf=self.config.verification.kinf)
        
        # Stage 2: Verify each response
        scores = self.verify_responses(responses, kverif=self.config.verification.kverif)
        
        # Find the top responses
        top_indices = self._find_top_responses(scores)
        
        # Stage 3: Compare top candidates if there's a tie
        if len(top_indices) > 1:
            winner_idx = self.compare_responses([responses[i] for i in top_indices], ktie=self.config.verification.ktie)
            best_response = responses[top_indices[winner_idx]]
        else:
            best_response = responses[top_indices[0]]
        
        return best_response
```
