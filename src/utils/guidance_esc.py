# src/utils/guidance_esc.py
# -*- coding: utf-8 -*-

"""
ESC (Emotional Support Conversation) guidance + strategy-chain summary generators.

对齐你 CB 版本的写法：
- 只用 call_plato_api（你已有）
- 不依赖 prompts/*.py
- prompt 直接写在代码里

用途：
- generate_high_level_guidance: 用“supporter”视角给出下一步高层指导（1–2句）
- generate_strategy_chain_summary: 总结检索到的策略链，给 supporter 长期规划方向（1–2句）
"""

from typing import List, Dict, Any, Optional
from src.utils.llm_api import call_plato_api  # 你已有的 API


def generate_high_level_guidance(
    principles: List[Dict[str, Any]],
    last_user_utt: str,   # ESC里这里是 seeker 最新一句（user-signal）
    recent_dialogue: str,
    emotion_type: Optional[str] = None,
    problem_type: Optional[str] = None,
    situation: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    max_tokens: int = 200
) -> str:
    """
    调用大模型，对 top-k principles + seeker 最新一句话 + 最近对话上下文进行总结，
    形成一条精炼的“高层支持策略指导”，用于指导 supporter 下一步如何回应/引导。

    Returns:
        一条字符串，高层支持策略指导（1-2 句）
    """
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
        out = call_plato_api(
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
    """
    调用大模型，对“成功支持轨迹”的策略链进行总结，生成 supporter 的长期规划方向（1–2 句）。

    Returns:
        一条字符串，长期规划指导（1-2 句）
    """
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
        out = call_plato_api(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (out or "").strip()
    except Exception as e:
        print(f"[warn] calling generate_strategy_chain_summary_esc failed: {e}")
        return ""