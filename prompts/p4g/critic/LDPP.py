# ===================== LDPP Critic Prompt =====================
from textwrap import dedent
from typing import Tuple


def build_critic_prompt_ldpp(dialogue_text: str) -> Tuple[str, str]:
    """
    Build LDPP prompt for Critic model based on Table 19.
    
    Returns:
        (system_prompt, user_prompt) tuple
    """
    system_prompt = dedent("""
    Given a conversation between a Persuader and a Persuadee, please assess whether the Persuadee has agreed to donate to Save the Children.
    """).strip()
    
    user_prompt = dedent(f"""
    The following is a conversation: [conversation]
    {dialogue_text}
    
    Question: Has the Persuadee agreed to donate to Save the Children?
    
    You can only answer using the following sentences:
    A) The persuadee has explicitly refused.
    B) The persuadee remains neutral about donating.
    C) The persuadee has a positive attitude towards donating but hasn't decided yet.
    D) The persuadee has decided to donate.
    
    Answer (one letter only):
    """).strip()
    
    return system_prompt, user_prompt

