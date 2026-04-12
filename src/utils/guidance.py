from typing import List, Dict, Any, Optional
from src.utils.llm_api import call_llm_api
from prompts.p4g.runtime_template import build_strategy_chain_summary_prompt

def generate_high_level_guidance(
    principles: List[Dict[str, Any]],
    last_user_utt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    max_tokens: int = 200
) -> str:

    if not principles:
        return ""

    principle_list_txt = "\n".join(
        [f"- {p.get('principle_text', '')}" for p in principles]
    )

    system_msg = """\
You are an expert persuasion strategist.

Your task:
- Read several micro-principles (which may describe conditions).
- Read the user's latest statement.
- Then provide ONE high-level strategy (1–2 sentences) that directly tells the persuader 
  what they should do next.
  
Requirements:
- Focus on the next action the persuader should take.
- Do NOT use "when..." or restate conditions from the principles.
- Do NOT copy or paraphrase the principles.
- Provide a direct, concise strategic action.

Example (DO NOT imitate content, only imitate format):
"1. Shift from abstract arguments to a relatable, real-world example that shows how their contribution makes a visible difference."
"2. Affirm their concerns, and connect your request to a broader positive outcome they would feel proud to support."
"""


    user_msg = f"""\
[User's Latest Message]
{last_user_utt}

[Relevant Micro-Principles]
{principle_list_txt}

Task:
Provide ONE high-level strategy (1–2 sentences) describing what the persuader 
should do next. Focus only on the next strategic action.
"""


    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        out = call_llm_api(
            messages, 
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return out.strip()
    except Exception as e:
        print(f"[warn] calling generate_high_level_guidance failed: {e}")
        return ""


def generate_strategy_chain_summary(
    strategy_chain_hint: str,
    current_turn_id: int,
    recent_dialogue: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    max_tokens: int = 200
) -> str:
    if not strategy_chain_hint or not strategy_chain_hint.strip():
        return ""
    
    system_prompt, user_prompt = build_strategy_chain_summary_prompt(
        strategy_chain_hint=strategy_chain_hint,
        current_turn_id=current_turn_id,
        recent_dialogue=recent_dialogue
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        out = call_llm_api(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return out.strip()
    except Exception as e:
        print(f"[warn] calling generate_strategy_chain_summary failed: {e}")
        return ""
