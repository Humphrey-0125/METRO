# prompts/cb/buyer/procot.py
# -*- coding: utf-8 -*-

from typing import Tuple
from textwrap import dedent


def build_buyer_prompt_procot(
    dialogue_text: str,
    item_name: str,
    buyer_target_price: float,
    item_description: str,
) -> Tuple[str, str]:
    """
    ProCoT Buyer prompt for CraigslistBargain (CB).

    Difference from Proactive:
    - Requires explicit structured reasoning before action:
      (progress analysis, decision on price move, next-turn goal).
    """

    # === system prompt: 与 standard / proactive 对齐 ===
    system_prompt = dedent(f"""
    You are a Buyer negotiating the price of a {item_name}.
    Your target price is {buyer_target_price}.

    Reply in 1–2 short, natural sentences.
    """).strip()

    # === user prompt: 明确“是否报价”的决策 + CoT 结构 ===
    user_prompt = dedent(f"""
    Item description:
    {item_description}

    Conversation so far:
    {dialogue_text}

    Write the Buyer's next reply using the EXACT format below.

    Buyer negotiation strategies (A–K):
    A greet
    B ask_question
    C answer_question
    D propose_first_price
    E propose_counter_price
    F use_comparatives
    G confirm_information
    H affirm_confirmation
    I deny_confirmation
    J agree_with_proposal
    K disagree_with_proposal

    Format (must follow exactly):
    Progress analysis: <max 2 sentences>
    Next-turn goal: <1 sentence>
    Which one is the most appropriate dialogue strategy: <A-K, comma-separated>
    Buyer response: <1-2 concise sentence>

    Do not output anything other than the required format.
    """).strip()

    return system_prompt, user_prompt