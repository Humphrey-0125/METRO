# ===================== LDPP Prompts (from paper tables) =====================
# This module provides a unified interface to access LDPP prompts
# The actual prompt functions are split into separate files:
# - persuader/LDPP.py: build_persuader_generation_prompt_ldpp
# - persuadee/LDPP.py: build_persuadee_generation_prompt_ldpp
# - critic/LDPP.py: build_critic_prompt_ldpp

from typing import Optional, Tuple

# Import from separate modules
from prompts.persuader.LDPP import build_persuader_generation_prompt_ldpp
from prompts.persuadee.LDPP import build_persuadee_generation_prompt_ldpp
from prompts.critic.LDPP import build_critic_prompt_ldpp

# Re-export for backward compatibility
__all__ = [
    'build_persuader_generation_prompt_ldpp',
    'build_persuadee_generation_prompt_ldpp',
    'build_critic_prompt_ldpp',
]

