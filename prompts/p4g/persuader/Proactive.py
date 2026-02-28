from typing import Tuple

def build_persuader_prompt_proactive(dialogue_text: str) -> Tuple[str, str]:
    system_prompt = (
        'You are a Persuader trying to persuade the Persuadee to donate to the charity '
        '"Save the Children", which helps children in poverty and war zones. '
        'Even small donations ($1–$2) can make a difference.'
    )

    user_prompt = f"""
Given the conversation history, select the most appropriate persuasion strategy letter(s)
from the list below, then write the Persuader’s next utterance.

Strategies:
A Logical appeal | B Emotion appeal | C Credibility appeal | D Foot-in-the-door
E Self-modeling | F Personal story | G Donation information
H Source-related inquiry | I Task-related inquiry | J Personal-related inquiry

Output format (must follow exactly):
Selected persuasion strategies: <A-J, comma-separated>
Persuader response: <one concise sentence>

Do not output anything other than the required format.

Conversation history:
{dialogue_text}
""".strip()

    return system_prompt, user_prompt
