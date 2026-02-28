from textwrap import dedent
from typing import Tuple


def build_buyer_generation_prompt_principles(
    dialogue_text: str,
    principles_block: str,
    item_name: str,
    buyer_target_price: str,
    item_description: str,
) -> Tuple[str, str]:
    """
    PRINCIPLES-guided buyer prompt for CraigslistBargain (CB).

    This buyer reacts locally to the seller's last move, while being
    implicitly guided by retrieved strategy principles.
    No explicit planning or strategy selection is required.
    """

    system_prompt = dedent(f"""
    You are a Buyer negotiating the price of a {item_name}.
    Your target price is ${buyer_target_price}.

    Reply in 1–2 short and succinct sentences.
    """).strip()


    user_prompt = dedent(f"""
    Item description:
    {item_description}

    Conversation so far:
    {dialogue_text}

    Strategy principles relevant to the current situation:
    [PRINCIPLES]
    {principles_block}
    [/PRINCIPLES]

    Write the Buyer's next reply (1-2 short and succinct sentence).
    Follow the principles implicitly, but do not mention them.
    """).strip()

    return system_prompt, user_prompt
