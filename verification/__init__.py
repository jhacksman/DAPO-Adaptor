# DAPO-Adaptor Verification Module

"""
This module implements the Sample, Scrutinize and Scale verification approach
for the DAPO framework.

The verification approach consists of three stages:
1. Generate multiple candidate responses
2. Verify each response
3. Compare top candidates to break ties

See the documentation in docs/verification/ for more details.
"""

from .verification_rollout import VerificationRollout
from .verification_utils import verify_response, compare_responses
from .prompts import VERIFICATION_PROMPT, COMPARISON_PROMPT

__all__ = [
    'VerificationRollout',
    'verify_response',
    'compare_responses',
    'VERIFICATION_PROMPT',
    'COMPARISON_PROMPT',
]
