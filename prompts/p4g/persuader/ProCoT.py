from typing import Tuple

def build_persuader_prompt_procot(dialogue_text: str) -> Tuple[str, str]:
    system_prompt = (
        'You are a Persuader trying to persuade the Persuadee to donate to the charity '
        '"Save the Children", which helps children in poverty and war zones. '
        'Even small donations ($1–$2) can make a difference.'
    )

    user_prompt = f"""
Given the conversation history, write ONE response with the exact format below.

Strategies (A–J):
A Logical | B Emotion | C Credibility | D Foot-in-the-door | E Self-modeling
F Personal story | G Donation info | H Source inquiry | I Task inquiry | J Personal inquiry

Format:
Progress analysis: <max 2 sentences>
Next-turn goal: <1 sentence>
Selected persuasion strategies: <A-J, comma-separated>
Persuader response: <one concise sentence>

Do not output anything other than the required format.

Conversation history:
{dialogue_text}
""".strip()

    return system_prompt, user_prompt
