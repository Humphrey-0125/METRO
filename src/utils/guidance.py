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
    """
    调用大模型，对 top-k principles + 当前 persuadee 的 utterance 
    进行总结，形成一条精炼的“高层策略指导”。

    Args:
        principles: top-k 检索得到的 principles 列表
        last_user_utt: 最近的用户一句话
        model: SiliconFlow 的 model 名
        temperature: 生成温度
        max_tokens: 最大输出 token

    Returns:
        一条字符串，高层策略指导（1-2 句）
    """

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
    """
    调用大模型，对长策略链进行总结，生成一个关于"如何长期规划"的指导。
    
    Args:
        strategy_chain_hint: 格式化的策略链文本（来自 format_strategy_hint）
        current_turn_id: 当前对话轮次
        recent_dialogue: 最近的对话历史（上一轮 Persuader 和 Persuadee 的对话），用于理解当前状况
        model: 使用的模型名称
        temperature: 生成温度
        max_tokens: 最大输出 token
        
    Returns:
        一条字符串，长期规划指导（1-2 句）
    """
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
