from textwrap import dedent
from typing import Optional, Tuple


def build_persuader_generation_prompt(
    dialogue_text: str,
    strategy_chain_hint: Optional[str] = None,
    guidance_text: Optional[str] = None,
    prompt_type: str = "Ours",
    use_depth: bool = True,       # ✅ NEW
    use_breadth: bool = True,     # ✅ NEW
) -> Tuple[str, str]:
    """
    Persuader prompt with optional depth (strategy chain) and breadth (high-level guidance).

    depth:
        - strategy_chain_hint / long-term planning
    breadth:
        - guidance_text / immediate response

    IMPORTANT:
    - If use_depth=True and use_breadth=True, behavior is IDENTICAL to the original implementation.
    - Ablation is structural: removed perspectives do NOT appear anywhere in the prompt.
    """

    # === system prompt（保持不变） ===
    system_prompt = dedent("""
    You are the Persuader in a persuasion conversation. 
    **Ultimate Goal:** 
    - Guide the Persuadee to make a donation to Save the Children as soon as it becomes appropriate.
    - The donation does not need to be large; even $1–$2 meaningfully helps children in need.
    - Avoid excessive questioning that stalls progress; questions should move closer to donation.
    """).strip()

    sections = [
        "[Conversation So Far]",
        dialogue_text,
    ]

    # ------------------------------------------------
    # Depth: Strategy chain / long-term planning
    # ------------------------------------------------
    if use_depth and strategy_chain_hint and strategy_chain_hint.strip():
        if prompt_type == "Ours_1":
            # Ours_1: summarized long-term planning
            sections.append(
                "\n[Long-Term Planning Summary]\n"
                "This summarizes the strategic approach from successful similar dialogues:\n"
                + strategy_chain_hint.strip()
            )
        else:
            # Ours: raw strategy chain
            sections.append(
                "\n[Strategy Chain Recommendation]\n"
                "These reflect long-term successful persuasion trajectories from similar dialogues:\n"
                + strategy_chain_hint.strip()
            )

    # ------------------------------------------------
    # Breadth: High-level guidance / immediate response
    # ------------------------------------------------
    if use_breadth and guidance_text and guidance_text.strip():
        sections.append(
            "\n[High-Level Guidance]\n"
            "This recommendation captures what is most effective for responding to the user's current state:\n"
            + guidance_text.strip()
        )

    # ------------------------------------------------
    # Instruction（严格区分消融状态）
    # ------------------------------------------------
    if prompt_type == "Ours_1":

        # === Full: depth + breadth（与原实现完全等价） ===
        if use_depth and use_breadth:
            sections.append(dedent("""
            [Instruction]
            Think from two perspectives:
            1. **Long-term planning** — use the long-term planning summary to understand the overall strategic direction and key phases of persuasion.
            2. **Immediate response** — use the high-level guidance to adapt sensitively to the user's latest message.

            Combine both perspectives to decide your next move.
            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

        # === w/o depth ===
        elif (not use_depth) and use_breadth:
            sections.append(dedent("""
            [Instruction]
            Focus on adapting sensitively to the user's latest message using the high-level guidance.

            Decide your next move accordingly.
            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

        # === w/o breadth ===
        elif use_depth and (not use_breadth):
            sections.append(dedent("""
            [Instruction]
            Focus on maintaining a coherent long-term persuasion trajectory based on the planning summary.

            Decide your next move accordingly.
            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

        # === w/o both ===
        else:
            sections.append(dedent("""
            [Instruction]
            Decide your next move naturally as a persuader in the conversation.

            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

    else:
        # -------- 非 Ours_1（保持对称） --------

        if use_depth and use_breadth:
            sections.append(dedent("""
            [Instruction]
            Think from two perspectives:
            1. Long-term planning — let the strategy-chain recommendations guide the overall persuasion direction.
            2. Immediate response — use the high-level guidance to adapt to the user's current state.

            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

        elif (not use_depth) and use_breadth:
            sections.append(dedent("""
            [Instruction]
            Focus on adapting to the user's current state using the high-level guidance.

            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

        elif use_depth and (not use_breadth):
            sections.append(dedent("""
            [Instruction]
            Let the strategy-chain recommendations guide the overall persuasion direction.

            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

        else:
            sections.append(dedent("""
            [Instruction]
            Respond naturally as a persuader in the conversation.

            Write the Persuader's next reply in 1–2 short, natural sentences only.
            """).strip())

    user_prompt = "\n\n".join([s for s in sections if s is not None])

    return system_prompt, user_prompt






def build_strategy_chain_summary_prompt(
    strategy_chain_hint: str,
    current_turn_id: int,
    recent_dialogue: Optional[str] = None
) -> Tuple[str, str]:
    """
    构建用于总结策略链的 prompt，使模型在当前情况下为 persuader 给出长期规划方向。
    """
    system_prompt = dedent("""
    You are an expert persuasion strategist specialized in long-term planning for the persuader.

    Your task:
    - Interpret the current persuasion context based on the recent dialogue.
    - Review 1 strategy chain extracted from successful persuasion trajectories.
    - Provide a high-level summary (1–2 sentences) that explains how the persuader should plan their long-term strategy
      given the current situation and the successful trajectory.

    Requirements:
    - The summary must clearly address what *the persuader* should aim to achieve over time.
    - Focus on strategic direction and psychological progression, not specific short-term actions.
    - Describe how the persuader’s approach should evolve (e.g., reducing doubt, building value alignment, fostering intrinsic motivation).
    - Do NOT list operational steps, give bullet lists, or restate explicit actions from the chain.
    - Connect the strategic trajectory to the current conversational state.

    Example style (do NOT copy wording):
    "Given the user’s hesitation, the persuader should focus on reducing uncertainty and gradually aligning the cause with the user’s personal motivations, so that donating later feels natural and self-driven."
    """).strip()

    sections = []
    if recent_dialogue and recent_dialogue.strip():
        sections.append(f"[Recent Conversation Context]\n{recent_dialogue.strip()}")

    sections.append(f"[Strategy Chain from Successful Dialogues]\n{strategy_chain_hint.strip()}")
    sections.append("""
Task:
Based on the current situation and the successful trajectory, provide a concise (1–2 sentences)
long-term strategic direction for the *persuader*, describing how their persuasion approach should evolve over time.
""".strip())

    user_prompt = "\n\n".join(sections)
    return system_prompt, user_prompt


