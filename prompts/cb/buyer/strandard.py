from textwrap import dedent
from typing import Tuple


def build_buyer_generation_prompt_standard(
    dialogue_text: str,
    item_name: str,
    buyer_target_price: float,
    item_description: str,
) -> Tuple[str, str]:
    """
    Standard (baseline) buyer prompt for CraigslistBargain (CB).

    This buyer reacts locally to the seller's last move without
    explicit planning or strategy selection.
    """

    system_prompt = dedent(f"""
    You are a Buyer negotiating the price of a {item_name}.
    Your target price is {buyer_target_price}.

    Reply in 1–2 short, natural sentences.
    """).strip()

    user_prompt = dedent(f"""
    Item description:
    {item_description}

    Conversation so far:
    {dialogue_text}

    Write the Buyer's next reply.
    """).strip()

    return system_prompt, user_prompt