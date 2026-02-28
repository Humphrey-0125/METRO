# ===================== LDPP Persuader Prompt =====================
from textwrap import dedent
from typing import Optional, Tuple


def build_persuader_generation_prompt_ldpp(
    dialogue_text: str,
    strategy_hint_text: Optional[str] = None
) -> Tuple[str, str]:
    """
    Build LDPP prompt for Persuader generation.
    
    Returns:
        (system_prompt, user_prompt) tuple
    """
    system_prompt = dedent("""
    Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children. Save the Children is headquartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
    """).strip()
    
    user_prompt = dedent(f"""
    [conversation]
    {dialogue_text}
    
    Please reply with only one short and succinct sentence.
    """).strip()
    
    return system_prompt, user_prompt

