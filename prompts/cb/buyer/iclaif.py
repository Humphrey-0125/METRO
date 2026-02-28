# prompts/cb/buyer/iclaif.py
# -*- coding: utf-8 -*-

from textwrap import dedent
from typing import Tuple


# ===================== ICL-AIF Buyer Prompt (CB) =====================

def build_coach_suggestion_prompt_cb_iclaif(dialogue_text: str) -> Tuple[str, str]:
    """
    ICL-AIF Coach (CB):
    Provide brief, tactical suggestions to help the buyer negotiate effectively.
    IMPORTANT: Do NOT propose any specific numeric prices.
    """

    system_prompt = dedent("""
    Now enter the role-playing mode. In the following conversation, you will play as a coach in a bargain game. There will be a buyer and a seller bargaining about a product price.
    """).strip()

    user_prompt = dedent(f"""
    Read the following conversation between the buyer and the seller, then give 3 suggestions to the buyer about how to negotiate more effectively and persuade the seller to accept a better price. Each suggestion should be only one short and succinct sentence. The following is the conversation:
    {dialogue_text}

    Question: What are your suggestions?
    Answer:
    """).strip()

    return system_prompt, user_prompt


def build_buyer_generation_prompt_cb_iclaif(
    dialogue_text: str,
    coach_suggestions: str,
    item_name: str,
    buyer_target_price: float,
    item_description: str,
) -> Tuple[str, str]:

    system_prompt = dedent(f"""
    You are a Buyer negotiating the price of a {item_name}.
    Your target price is {buyer_target_price}.

    Rules:
    - Reply in 1-2 natural sentences.
    - You MUST use the coach's advice to decide your next move.
    """).strip()

    user_prompt = dedent(f"""
    Item description:
    {item_description}

    Conversation so far:
    {dialogue_text}

    Coach suggestions:
    {coach_suggestions}

    Write ONLY the buyer's next reply (1-2 sentences).
    """).strip()

    return system_prompt, user_prompt