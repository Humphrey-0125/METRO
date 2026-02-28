# ===================== LDPP Persuadee Prompt =====================
from textwrap import dedent
from typing import Tuple


def build_persuadee_generation_prompt_ldpp(dialogue_text: str) -> Tuple[str, str]:
    """
    Build LDPP prompt for Persuadee (User simulation) based on Table 17.
    
    Returns:
        (system_prompt, user_prompt) tuple
    """
    system_prompt = dedent("""
        You are a Persuader trying to persuade the Persuadee to donate to the charity 
        Save the Children", which helps children in poverty and war zones. 
        Even small donations ($1–$2) can make a difference.
    """).strip()

    user_prompt = f"""
    Conversation:
    {dialogue_text}

    Write ONLY the Persuader's next utterance (one concise sentence).
    """
    
    return system_prompt, user_prompt

