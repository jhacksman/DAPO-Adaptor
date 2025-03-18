# Integration Points

This document identifies the specific integration points in the DAPO codebase where the verification approach will be integrated.

## Key Files

The following files will need to be modified or extended:

1. **vLLMRollout Class**: The main class that handles response generation.
   - File: `~/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`
   - Purpose: Extend this class to add verification capabilities.

2. **BaseRollout Class**: The base class for all rollout implementations.
   - File: `~/verl/verl/workers/rollout/base.py`
   - Purpose: Understand the interface that all rollout implementations must follow.

3. **RayPPOTrainer Class**: The main trainer class that uses the rollout.
   - File: `~/verl/verl/trainer/ppo/ray_trainer.py`
   - Purpose: Understand how the rollout is used in the training process.

## Integration Strategy

### 1. Create a New Verification Module

Create a new module in the DAPO-Adaptor repository that implements the verification approach:

```
~/DAPO-Adaptor/verification/
├── __init__.py
├── verification_rollout.py
├── verification_utils.py
└── prompts.py
```

### 2. Extend the vLLMRollout Class

Create a new `VerificationRollout` class that extends the `vLLMRollout` class:

```python
# ~/DAPO-Adaptor/verification/verification_rollout.py
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from verl import DataProto

class VerificationRollout(vLLMRollout):
    """A rollout that implements the Sample, Scrutinize and Scale verification approach."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize verification parameters
        
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences with verification."""
        # Implement the three-stage verification process
        # ...
```

### 3. Create Verification Utilities

Create utility functions for the verification process:

```python
# ~/DAPO-Adaptor/verification/verification_utils.py
from typing import List
from verl import DataProto

def verify_response(response: DataProto, model, kverif: int) -> float:
    """Verify a single response and return a verification score."""
    # ...
    
def compare_responses(response1: DataProto, response2: DataProto, model, ktie: int) -> int:
    """Compare two responses and return the index of the winner (0 or 1)."""
    # ...
```

### 4. Define Verification Prompts

Create a file with the prompts used for verification:

```python
# ~/DAPO-Adaptor/verification/prompts.py

VERIFICATION_PROMPT = """
Please verify if the following response to the question is correct.

Question: {question}

Response: {response}

Is this response correct? Return 1 if correct, 0 if incorrect.
"""

COMPARISON_PROMPT = """
Please compare the following two responses to the question and determine which one is correct.

Question: {question}

Response 1: {response1}

Response 2: {response2}

Which response is correct? Return 1 if Response 1 is correct, 2 if Response 2 is correct.
"""
```

### 5. Update Configuration

Define the configuration parameters for the verification approach:

```python
# Example configuration in a YAML file
verification:
  enabled: true
  kinf: 200  # Number of candidate responses to generate
  kverif: 50  # Number of verification samples per response
  ktie: 100  # Number of comparison samples per pair
  threshold: 0.05  # Threshold for considering responses as tied
```
