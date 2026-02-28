from textwrap import dedent
from typing import Tuple

def build_seeker_prompt_persona(
    dialogue_text: str,
    persona_description: str,
    emotion_type: str,
    problem_type: str,
) -> Tuple[str, str]:
    """
    ESConv seeker(patient) prompt with persona + dialogue history.

    Args:
        dialogue_text: history text, e.g., "Therapist: ...\nPatient: ...\n"
        persona_description: persona string for the patient
        emotion_type: e.g., "anxiety"
        problem_type: e.g., "ongoing depression"
        situation: optional free-text scene description (can be empty)
    """

    system_prompt = dedent("""
    Now enter the role-playing mode. In the following conversation, you will play as a patient in a counselling conversation with a therapist.
    """).strip()

    user_prompt = dedent(f"""
    You are the patient who is looking for the help from the simulator(therapist), because you have the emotional issue about {emotion_type} regarding {problem_type}.
    Act according to the following personality description:

    [Persona Description]
    {persona_description}

    Important:
    - Reply as the seeker based on your personality, emotions, and thinking style.
    - Stay consistent with your emotional issue (emotion_type) and problem context (problem_type).
    - If you think your are convinced by the simulator, you can say "solve the problem".

    The following is the conversation so far:
    [conversation]
    {dialogue_text}
    [/conversation]

    Please reply with only one short and succinct sentence.
    [seeker]:
    """).strip()

    return system_prompt, user_prompt