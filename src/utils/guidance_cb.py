# src/utils/guidance_cb.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from src.utils.llm_api import call_llm_api


def generate_high_level_guidance(
    principles: List[Dict[str, Any]],
    last_user_utt: str,
    recent_dialogue: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    max_tokens: int = 200
) -> str:
    if not principles:
        return ""

    principle_list_txt = "\n".join(
        [f"- {p.get('principle_text', '')}" for p in principles]
    )

    system_msg = """\
You are a negotiation coach for Craigslist-style bargaining.

Task:
- Read several micro-principles.
- Read the seller’s latest message.
- Output ONE high-level strategy (1–2 sentences) telling the buyer what to do next to improve the price.

Guidelines:
- Focus on a single next strategic move, not dialogue text.
- Do NOT restate or paraphrase the principles.
- Do NOT write the buyer’s reply.
- Keep it concise, direct, and actionable.

Example style (do NOT copy content):
"Probe flexibility with one targeted question, then anchor a conditional counteroffer tied to a concrete constraint."
""".strip()


    user_msg = f"""\
[Seller's Latest Message]
{last_user_utt}

[Recent Conversation Context]
{recent_dialogue}

[Relevant Micro-Principles]
{principle_list_txt}

Task:
Provide ONE high-level strategy (1–2 sentences) describing what the buyer should do next.
Focus only on the next strategic action.
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
        return out.strip()
    except Exception as e:
        print(f"[warn] calling generate_high_level_guidance_cb failed: {e}")
        return ""


def generate_strategy_chain_summary(
    strategy_chain_hint: str,
    recent_dialogue: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    max_tokens: int = 200
) -> str:
    if not strategy_chain_hint or not strategy_chain_hint.strip():
        return ""

    system_prompt = """\
You are an expert negotiation strategist for the buyer.

Task:
- Read the recent bargaining context.
- Review 1 strategy chain from successful similar negotiations.
- Produce a 1–2 sentence summary describing how the buyer should plan their negotiation over time.

Guidelines:
- Focus on long-term strategic direction and negotiation dynamics, not immediate actions.
- Do NOT list steps or bullet points.
- Do NOT restate actions verbatim from the strategy chain.
- Tie the strategy clearly to the current conversational state.

Example style (do NOT copy content):
"Given the seller’s firmness, the buyer should probe constraints to build leverage, then anchor a justified counteroffer and concede only conditionally toward a favorable deal."
""".strip()


    sections = []
    if recent_dialogue and recent_dialogue.strip():
        sections.append(f"[Recent Conversation Context]\n{recent_dialogue.strip()}")

    sections.append(f"[Strategy Chain from Successful Dialogues]\n{strategy_chain_hint.strip()}")

    sections.append(f"""\
Task:
Based on the recent conversation context and the successful trajectory,
provide a concise (1–2 sentences) long-term strategic direction for the buyer,
describing how the buyer's negotiation approach should evolve over time.
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
        return out.strip()
    except Exception as e:
        print(f"[warn] calling generate_strategy_chain_summary_cb failed: {e}")
        return ""
