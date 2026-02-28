# ===================== Standard Persuader Prompt =====================
from textwrap import dedent
from typing import Optional, Tuple


def build_simulator_prompt_standard(
    dialogue_text: str,
    strategy_hint_text: Optional[str] = None
) -> Tuple[str, str]:
    """
    Build Standard prompt for Persuader generation.
    
    Returns:
        (system_prompt, user_prompt) tuple
    """
    system_prompt = dedent("""
    Now enter the role-playing mode. You are the simulator who is trying to help the seeker(patient) reduce their emotional distress and help them understand and work through the challenges. 
    """).strip()
    
    user_prompt = dedent(f"""
    [conversation]
    {dialogue_text}
    
    Please reply with only one short and succinct sentence.
    [simulator]:
    """).strip()
    
    return system_prompt, user_prompt

