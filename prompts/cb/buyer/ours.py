# prompts/cb/buyer/ours.py
# -*- coding: utf-8 -*-

from textwrap import dedent
from typing import Optional, Tuple


def build_buyer_generation_prompt_cb_ours(
    dialogue_text: str,
    item_name: str,
    buyer_target_price: float,
    item_description: str,
    strategy_chain_hint: Optional[str] = None,
    guidance_text: Optional[str] = None,
    prompt_type: str = "Ours",
    use_depth: bool = True,       # 👈 新增
    use_breadth: bool = True,     # 👈 新增
) -> Tuple[str, str]:
    """
    CB buyer prompt with optional two-perspective guidance.

    depth (long-term planning):
        - strategy_chain_hint / summary
    breadth (immediate response):
        - guidance_text

    IMPORTANT:
    - If use_depth=True and use_breadth=True, Ours_1 behavior is IDENTICAL to the original version.
    - Ablation is structural: removed perspectives do not appear anywhere in the prompt.
    """

    # === system prompt（保持不变） ===
    system_prompt = dedent(f"""
    You are a Buyer negotiating the price of a {item_name}.
    Your target price is {buyer_target_price}.

    Reply in 1–2 natural sentences.
    """).strip()

    sections = [
        "[Item description]",
        (item_description or "").strip(),
        "",
        "[Conversation So Far]",
        (dialogue_text or "").strip(),
    ]

    # ------------------------------------------------
    # Depth: Strategy chain / long-term planning
    # ------------------------------------------------
    if use_depth and strategy_chain_hint and str(strategy_chain_hint).strip():
        if (prompt_type or "").strip() == "Ours_1":
            sections.append(
                "\n[Long-Term Planning Summary]\n"
                "This summarizes successful bargaining trajectories in similar dialogues:\n"
                + str(strategy_chain_hint).strip()
            )
        else:
            sections.append(
                "\n[Strategy Chain Recommendation]\n"
                "These reflect effective long-term bargaining trajectories from similar dialogues:\n"
                + str(strategy_chain_hint).strip()
            )

    # ------------------------------------------------
    # Breadth: High-level guidance / immediate response
    # ------------------------------------------------
    if use_breadth and guidance_text and str(guidance_text).strip():
        sections.append(
            "\n[High-Level Guidance]\n"
            "This captures the most effective response to the seller's latest message:\n"
            + str(guidance_text).strip()
        )

    # ------------------------------------------------
    # Instruction（根据消融状态自动匹配）
    # ------------------------------------------------
    if (prompt_type or "").strip() == "Ours_1":

        # === 完整版本（严格等价原 Ours_1） ===
        if use_depth and use_breadth:
            sections.append(dedent("""
            [Instruction]
            Think from two perspectives:
            1) Long-term planning — use the long-term planning summary to maintain a good bargaining trajectory.
            2) Immediate response — use the high-level guidance to respond appropriately to the seller’s latest message.

            Write the Buyer's next reply in 1–2 short, natural sentences only.
            """).strip())

        # === w/o depth ===
        elif (not use_depth) and use_breadth:
            sections.append(dedent("""
            [Instruction]
            Focus on responding appropriately to the seller’s latest message using the high-level guidance.

            Write the Buyer's next reply in 1–2 short, natural sentences only.
            """).strip())

        # === w/o breadth ===
        elif use_depth and (not use_breadth):
            sections.append(dedent("""
            [Instruction]
            Focus on maintaining a good long-term bargaining trajectory based on the long-term planning summary.

            Write the Buyer's next reply in 1–2 short, natural sentences only.
            """).strip())

        # === w/o both ===
        else:
            sections.append(dedent("""
            [Instruction]
            Respond naturally as a buyer in the negotiation.

            Write the Buyer's next reply in 1–2 short, natural sentences only.
            """).strip())

    user_prompt = "\n\n".join([s for s in sections if s is not None])

    return system_prompt, user_prompt