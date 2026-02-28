from typing import Tuple
from textwrap import dedent


def build_buyer_prompt_proactive(
    dialogue_text: str,
    item_name: str,
    buyer_target_price: float,
    item_description: str,
) -> Tuple[str, str]:
    """
    Proactive Buyer prompt for CraigslistBargain (CB).

    Buyer explicitly decides whether proposing a new price is necessary
    before selecting a negotiation strategy.
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

    Buyer negotiation strategies:
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

    Output format (must follow exactly):
    Which one is the most appropriate dialogue strategy: <A-K, comma-separated>
    Buyer response: <1-2 concise sentence>

    Do not output anything else.
    """).strip()

    return system_prompt, user_prompt