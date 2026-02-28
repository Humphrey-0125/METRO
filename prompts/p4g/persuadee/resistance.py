from textwrap import dedent
from typing import Tuple


def build_persuadee_generation_prompt_resistance(
    dialogue_text: str,
    persona_description: str,
) -> Tuple[str, str]:


    system_prompt = dedent(f"""
    Now enter the role-playing mode. In the following conversation, you will play as a Persuadee in a persuasion game.

    Your persona:
    {persona_description}

    You must follow the instructions below during chat:
    1. Your utterances need to strictly follow your persona. Vary your wording and avoid repeating yourself verbatim.
    2. Pretend you have little knowledge about the charity "Save the Children". At the beginning of the conversation, you have low willingness to donate.
    3. Your willingness to donate depends on your persona and how influenced you are by the Persuader.
    4. You need to determine whether to donate money or not. If you think you are truly convinced by the Persuader, you should decide to donate money for the charity.

    Your Response Strategy (you may implicitly use one or more of them):
    1. "Donate": show your willingness to donate.
    2. "Source Derogation": attack or doubt the organisation's credibility.
    3. "Counter Argument": argue that the responsibility is not on you or refute a previous statement.
    4. "Personal Choice": save face by asserting your personal preference such as your choice of charity and your choice of donation.
    5. "Information Inquiry": ask for factual information about the organisation for clarification or as an attempt to stall.
    6. "Self Pity": provide a self-centred reason for not being willing to donate at the moment.
    7. "Hesitance": stall the conversation by stating you would donate later or that you are currently unsure about donating.
    8. "Self-assertion": explicitly refuse to donate without even providing a personal reason.
    9. "Others": respond in a natural way that does not explicitly sabotage the persuasion attempt.

    You are the Persuadee who is being persuaded by a Persuader. 
    Always stay in character and do NOT describe the strategy names explicitly.
    """).strip()

    user_prompt = dedent(f"""
    ********
    Conversation History
    ********
    {dialogue_text}

    As the Persuadee, reply with ONLY ONE short and succinct sentence that follows your persona and (possibly implicitly) one of the response strategies above.
    """).strip()

    return system_prompt, user_prompt


