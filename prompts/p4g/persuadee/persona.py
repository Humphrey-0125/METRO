from textwrap import dedent
from typing import Tuple

def build_persuadee_generation_prompt_persona(
    dialogue_text: str,
    persona_description: str
) -> Tuple[str, str]:
    """
    Build persona-based persuadee simulation prompt.
    persona_description 是从 personas_eval.jsonl 读到的 description 字段。
    """

    system_prompt = dedent(f"""
    You are now entering role-playing mode.

    You are the Persuadee in a persuasion dialogue. 
    Act according to the following personality description:

    [Persona Description]
    {persona_description}

    Important:
    - You reply based on your personality, your emotions, and your thinking style.
    - Your willingness for donation depends on your persona and how influenced you are by the Persuader.
    - You need to determine whether to donate money or not. If you think your are convinced by the
    Persuader, you should donate money for the charity.
    """).strip()

    user_prompt = dedent(f"""
    [conversation]
    {dialogue_text}

    Please reply with ONE short sentence.
    """).strip()

    return system_prompt, user_prompt
