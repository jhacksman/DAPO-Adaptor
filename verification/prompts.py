# DAPO-Adaptor Verification Prompts

"""
This module provides prompt templates for the verification process,
including prompts for verifying responses and comparing responses.
"""

# Prompt for verifying a response
VERIFICATION_PROMPT = """
Please verify if the following response to the question is correct.

Question: {question}

Response: {response}

Is this response correct? Return 1 if correct, 0 if incorrect.
"""

# Prompt for comparing two responses
COMPARISON_PROMPT = """
Please compare the following two responses to the question and determine which one is correct.

Question: {question}

Response 1: {response1}

Response 2: {response2}

Which response is correct? Return 1 if Response 1 is correct, 2 if Response 2 is correct.
"""

# Prompt for generating multiple responses
GENERATION_PROMPT = """
{question}
"""

# Additional prompts for specific verification tasks

# Prompt for mathematical verification
MATH_VERIFICATION_PROMPT = """
Please verify if the following mathematical solution is correct.

Problem: {question}

Solution: {response}

Is this solution correct? Return 1 if correct, 0 if incorrect.
Please show your work step by step to verify the solution.
"""

# Prompt for logical verification
LOGIC_VERIFICATION_PROMPT = """
Please verify if the following logical reasoning is correct.

Problem: {question}

Reasoning: {response}

Is this reasoning correct? Return 1 if correct, 0 if incorrect.
Please identify any logical fallacies or errors in the reasoning.
"""

# Prompt for factual verification
FACTUAL_VERIFICATION_PROMPT = """
Please verify if the following factual statement is correct.

Question: {question}

Statement: {response}

Is this statement factually correct? Return 1 if correct, 0 if incorrect.
Please provide evidence or counterevidence for your verification.
"""
