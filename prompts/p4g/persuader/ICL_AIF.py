# ===================== ICL-AIF Persuader Prompt (from Table 27) =====================
from textwrap import dedent
from typing import Tuple


def build_coach_suggestion_prompt_iclaif(dialogue_text: str) -> Tuple[str, str]:
    """
    Build ICL-AIF prompt for Coach to provide suggestions to the Persuader.
    Based on Section 1 of Table 27.
    
    Returns:
        (system_prompt, user_prompt) tuple
    """
    system_prompt = dedent("""
    Now enter the role-playing mode. In the following conversation, you will play as a coach in a persuasion game. There will be a persuader who is trying to persuade a persuadee for charity donation. Your task is to read the conversation between the persuader and the persuadee, then provide suggestions to the persuader about how to convince the persuadee to make a donation.
    """).strip()
    
    user_prompt = dedent(f"""
    Read the following conversation between the persuader and the persuadee, then give three suggestions to the persuader about how to convince the persuadee to make a donation. Each suggestion should be only one short and succinct sentence. The following is the conversation: [conversation]
    {dialogue_text}
    
    Question: What are your suggestions?
    Answer:
    """).strip()
    
    return system_prompt, user_prompt


def build_persuader_generation_prompt_iclaif(dialogue_text: str, coach_suggestions: str) -> Tuple[str, str]:
    system_prompt = (
        'You are a Persuader trying to persuade the Persuadee to donate to the charity '
        '"Save the Children", which helps children in poverty and war zones. '
        'Even small donations ($1–$2) can make a difference.'
    )

    user_prompt = f"""
Conversation:
{dialogue_text}

Suggestions:
{coach_suggestions}

Write ONLY the Persuader's next utterance (one concise sentence).
""".strip()

    return system_prompt, user_prompt
