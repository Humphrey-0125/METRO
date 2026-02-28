from textwrap import dedent
from typing import Tuple

def build_esc_judge_prompt(
    emotion_type: str,
    problem_type: str,
    dialogue_text: str
) -> Tuple[str, str]:
    """
    ESConv critic prompt (match the table, no scene).
    Must answer with ONE of A/B/C/D sentences.
    """

    system_prompt = dedent("""
    Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation.
    """).strip()

    emotion_type = (emotion_type or "").strip() or "N/A"
    problem_type = (problem_type or "").strip() or "N/A"

    user_prompt = dedent(f"""
    You can only reply with one of the following sentences:
    A. No, the Patient feels worse.
    B. No, the Patient feels the same.
    C. No, but the Patient feels better.
    D. Yes, the Patient’s issue has been solved.

    If you believe that the patient’s problem has been resolved or the patient has realized how to solve the problem, please choose D.
    If you believe that the patient’s problem has not been fully resolved, but his emotional issue has been somewhat alleviated compared to the last conversation turn, you can choose C.
    If you believe that the patient’s emotional state has worsened compared to the last conversation turn, you can choose A.
    Otherwise, if the patient’s emotional state remains unchanged, please choose B.

    The following is a conversation about {emotion_type} regarding {problem_type}:
    {dialogue_text}
    Question: Has the Patient’s issue been solved?
    Answer:
    """).strip()

    return system_prompt, user_prompt