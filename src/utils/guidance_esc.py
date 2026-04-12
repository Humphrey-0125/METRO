# src/utils/guidance_esc.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from src.utils.llm_api import call_llm_api


def generate_high_level_guidance(
    principles: List[Dict[str, Any]],
    last_user_utt: str,
    recent_dialogue: str,
    emotion_type: Optional[str] = None,
    problem_type: Optional[str] = None,
    situation: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    max_tokens: int = 200
) -> str:
    if not principles:
        return ""

    principle_list_txt = "\n".join([f"- {p.get('principle_text', '')}" for p in principles])

    system_msg = """\
You are a clinical-style emotional support coach for ESC dialogues.

Task:
- Read several micro-principles for emotional support.
- Read the Seeker’s latest message and recent dialogue context.
- Output ONE high-level strategy (1–2 sentences) describing what the Supporter should do NEXT.

Guidelines:
- Do NOT write the Supporter’s actual reply text.
- Keep it concise (1–2 sentences), calm, and supportive.
""".strip()

    meta_lines = []
    if emotion_type:
        meta_lines.append(f"emotion_type: {emotion_type}")
    if problem_type:
        meta_lines.append(f"problem_type: {problem_type}")
    if situation:
        meta_lines.append(f"situation: {situation}")

    meta_block = ("\n".join(meta_lines)).strip()
    if meta_block:
        meta_block = f"[Meta]\n{meta_block}\n\n"

    user_msg = f"""\
{meta_block}[Seeker's Latest Message]
{last_user_utt}

[Recent Conversation Context]
{recent_dialogue}

[Relevant Micro-Principles]
{principle_list_txt}

Task:
Provide ONE high-level strategy (1–2 sentences) describing what the Supporter should do next.
Focus only on the next strategic support action (not the actual wording).
""".strip()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        out = call_llm_api(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (out or "").strip()
    except Exception as e:
        print(f"[warn] calling generate_high_level_guidance_esc failed: {e}")
        return ""


def generate_strategy_chain_summary(
    strategy_chain_hint: str,
    recent_dialogue: Optional[str] = None,
    emotion_type: Optional[str] = None,
    problem_type: Optional[str] = None,
    situation: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    max_tokens: int = 200
) -> str:
    if not strategy_chain_hint or not strategy_chain_hint.strip():
        return ""

    system_prompt = """\
You are an expert emotional support strategist (ESC).

Task:
- Read the recent dialogue context.
- Review ONE strategy chain from successful emotional-support conversations.
- Produce a 1–2 sentence summary describing how the Supporter should plan their support over time.

Guidelines:
- Do NOT list steps or bullet points.
- Do NOT restate the strategy chain verbatim.
- Tie the plan to the Seeker’s current emotional state and constraints.
- Keep it concise (1–2 sentences).
""".strip()

    sections = []

    meta_lines = []
    if emotion_type:
        meta_lines.append(f"emotion_type: {emotion_type}")
    if problem_type:
        meta_lines.append(f"problem_type: {problem_type}")
    # if situation:
    #     meta_lines.append(f"situation: {situation}")
    if meta_lines:
        sections.append("[Meta]\n" + "\n".join(meta_lines))

    if recent_dialogue and recent_dialogue.strip():
        sections.append(f"[Recent Conversation Context]\n{recent_dialogue.strip()}")

    sections.append(f"[Strategy Chain from Successful Dialogues]\n{strategy_chain_hint.strip()}")

    sections.append("""\
Task:
Based on the recent context and the successful trajectory,
provide a concise (1–2 sentences) long-term strategic direction for the Supporter,
describing how the support approach should evolve over time.
""".strip())

    user_prompt = "\n\n".join(sections)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        out = call_llm_api(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (out or "").strip()
    except Exception as e:
        print(f"[warn] calling generate_strategy_chain_summary_esc failed: {e}")
        return ""