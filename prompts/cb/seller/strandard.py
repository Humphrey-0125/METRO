from textwrap import dedent
from typing import Tuple


def build_seller_generation_prompt_standard(
    dialogue_text: str,
    item_name: str,
    seller_target_price: float,
    item_description: str,
) -> Tuple[str, str]:
    """
    Standard (baseline) seller prompt for CraigslistBargain (CB).

    This seller reacts locally to the buyer's last move without
    explicit planning or strategy selection.
    """

    system_prompt = dedent(f"""
    Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game.
    """).strip()

    user_prompt = dedent(f"""
    You are the seller who is trying to sell the {item_name} with the price of {seller_target_price}. 
    Product description:
    {item_description}

    Conversation so far:
    {dialogue_text}

    Write the Seller's next reply(one short, natural sentence).
    """).strip()

    return system_prompt, user_prompt
