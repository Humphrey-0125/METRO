from textwrap import dedent
from typing import Tuple


def build_seller_generation_prompt_persona(
    dialogue_text: str,
    persona_description: str,
    item_name: str,
    seller_target_price: float,
    item_description: str,
) -> Tuple[str, str]:
    """
    Seller simulation prompt for CraigslistBargain (CB).

    Seller acts as a rational, price-protective negotiator.
    The goal is to provide a stable and challenging negotiation environment
    to evaluate buyer negotiation ability.
    """

    system_prompt = dedent(f"""
    You are the Seller in a price bargaining game.

    Persona:
    {persona_description}

    You are selling a {item_name}.
    Your target price is {seller_target_price}.
    You prefer to sell this item at this price.

    Style rules:
    - Reply in 1-2 sentence.
    - Your utterances and bargain behavior need to strictly follow your persona. Varying your wording
    and avoid repeating yourself verbatim.
    - You can decide to change your target price flexibly based on your persona and the conversation.
    """).strip()

    user_prompt = dedent(f"""
    Item description:
    {item_description}

    Conversation so far:
    {dialogue_text}

    Write the Seller's next reply.
    """).strip()

    return system_prompt, user_prompt