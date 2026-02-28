# from textwrap import dedent
# from typing import Tuple

# def build_cb_judge_prompt(dialogue_text: str) -> Tuple[str, str]:
#     """
#     Alignment-style CB judge.
#     Output STRICT JSON only.
#     """
#     system_prompt = dedent("""
#     You are a strict judge for a price bargaining conversation between a Buyer and a Seller.

#     Your job is to extract the final bargaining positions of both sides and decide if a deal is reached.

#     Key idea:
#     A deal is reached ONLY when there exists a numeric price p such that:
#     - The Buyer explicitly accepts p, AND
#     - The Seller explicitly accepts p.
#     Otherwise, deal=false.

#     Decision:
#     - deal=true only if buyer_accepted_price and seller_accepted_price are both not null and equal (same numeric value).
#     - If deal=false, price must be null.

#     Output STRICT JSON exactly with these keys:
#     {
#       "deal": true/false,
#       "price": number|null,
#     }

#     Output ONLY JSON. No extra text.
#     """).strip()

#     user_prompt = dedent(f"""
#     [conversation]
#     {dialogue_text}
#     """).strip()

#     return system_prompt, user_prompt

from textwrap import dedent
from typing import Tuple

def build_cb_judge_prompt(dialogue_text: str) -> Tuple[str, str]:
    """
    Strict CB judge.
    Output STRICT JSON only.
    """

    system_prompt = dedent("""
    You are a strict judge for a price negotiation between a Buyer and a Seller.

    Your task: decide whether a deal is reached AT THE END of the conversation.

    IMPORTANT: Use a STRICT definition of "deal":
    - deal=true ONLY IF there is an explicit acceptance/confirmation to finalize the deal.
    - The Buyer must clearly accept a specific price (a number) or explicitly accept the Seller's last stated price.
    - If the conversation ends with a counteroffer, a final offer, a question, negotiation, or "does that work?",
      then deal=false.

    Price extraction rules:
    - If deal=true, output the agreed numeric price.
    - If multiple prices appear, choose the one that is explicitly accepted in the final agreement.
    - If deal=false, price must be null.

    Output STRICT JSON only:
    {"deal": true/false, "price": number or null}

    Do not output any other text.
    """).strip()

    user_prompt = dedent(f"""
    Decide if a deal is reached at the end of the conversation and extract the final agreed price if so.

    Examples:

    Conversation:
    Buyer: Can you do 630 for both pieces?
    Seller: I can meet you at 635 for both pieces.
    Question: Have they reached a deal at the end?
    Answer:
    {{"deal": false, "price": null}}

    Conversation:
    Buyer: Would you take 350?
    Seller: I can do 375.
    Buyer: Okay, 375 works for me. Let's do it.
    Question: Have they reached a deal at the end?
    Answer:
    {{"deal": true, "price": 375}}

    Conversation:
    Buyer: I can do 1650.
    Seller: 1650 works. Deal.
    Question: Have they reached a deal at the end?
    Answer:
    {{"deal": true, "price": 1650}}

    Conversation:
    Buyer: Could we meet halfway at 1630?
    Seller: I can reluctantly agree to 1650 as a final offer.
    Question: Have they reached a deal at the end?
    Answer:
    {{"deal": false, "price": null}}

    Now judge this conversation:
    {dialogue_text}

    Answer:
    """).strip()

    return system_prompt, user_prompt
