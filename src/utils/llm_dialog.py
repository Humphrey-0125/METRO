from typing import List, Dict, Any, Optional, Union
from src.utils.llm_api import call_llm_api
import re

def chat_completion(messages, model: str = None, temperature: float = 0.3,max_tokens: int = None) -> str:
    """通用对话生成函数。"""
    if model is None:
        return call_llm_api(messages, temperature=temperature)
    else:
        return call_llm_api(messages, model=model, temperature=temperature,max_tokens=max_tokens )

def chat_completion_persuader(messages, model: str = None, temperature: float = 0.3,max_tokens: int = None) -> str:
    """通用对话生成函数（这里用于 Persuader/Persuadee 发言）。"""
    if model is None:
        return call_llm_api(messages, temperature=temperature)
    else:
        return call_llm_api(messages, model=model, temperature=temperature,max_tokens=max_tokens)

def chat_completion_persuadee(messages, model: str = None, temperature: float = 0.3,max_tokens: int = None) -> str:
    """通用对话生成函数（这里用于 Persuader/Persuadee 发言）。"""
    if model is None:
        return call_llm_api(messages, temperature=temperature)
    else:
        return call_llm_api(messages, model=model, temperature=temperature,max_tokens=max_tokens)

def chat_completion_buyer(messages, model: str = None, temperature: float = 0.3,max_tokens: int = None) -> str:
    """通用对话生成函数（这里用于 Persuader/Persuadee 发言）。"""
    if model is None:
        return call_llm_api(messages, temperature=temperature)
    else:
        return call_llm_api(messages, model=model, temperature=temperature,max_tokens=max_tokens)

def chat_completion_seller(messages, model: str = None, temperature: float = 0.3,max_tokens: int = None) -> str:
    """通用对话生成函数（这里用于 Persuader/Persuadee 发言）。"""
    if model is None:
        return call_llm_api(messages, temperature=temperature)
    else:
        return call_llm_api(messages, model=model, temperature=temperature,max_tokens=max_tokens)

def history_to_plain_text(dialog_history: List[Dict[str, Any]]) -> str:
    """
    把对话历史拼成纯文本，保证每行只出现一次 speaker 前缀。
    会自动去除 text 中重复出现的 'Speaker:' 前缀（仅限行首）。
    """
    lines = []

    for turn in dialog_history:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        prefix_pattern = rf"^\s*{re.escape(speaker)}\s*:\s*"

        text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)

        lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


def _render_step(step: str) -> str:
    """
    将单个策略步骤转换为更易读的字符串：
    - 对于形如 "[a, b]" 的并列策略，转成 "a + b (并行)"；
    - 其他情况保持原样。
    """
    if not isinstance(step, str):
        return str(step)
    stripped = step.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if inner:
            items = [token.strip() for token in inner.split(",") if token.strip()]
            if len(items) > 1:
                return " + ".join(items)
            elif len(items) == 1:
                return items[0]
    return stripped


def format_strategy_hint(chains: Union[List[List[str]], List[List[List[str]]], List[Dict[str, Any]]], depth: Optional[int] = None) -> Optional[str]:
    """
    将多个 strategy_chain 格式化成一段可读的提示文本，喂给 Persuader。

    支持三种输入格式：
    - List[List[str]]: 旧格式，每个策略链是字符串列表
    - List[List[List[str]]]: 新格式，每个策略链是嵌套的策略列表
    - List[Dict[str, Any]]: 最新格式，每个元素是包含"chain"键的字典

    参数:
    - depth: 可选参数，控制只显示前几层的策略。如果为None则显示全部层
    """
    if not chains:
        return None

    formatted = []
    for idx, chain_item in enumerate(chains, start=1):
        if isinstance(chain_item, dict):
            chain = chain_item.get("chain", chain_item)
        else:
            chain = chain_item
        
        if depth is not None and chain:
            chain = chain[:depth]

        if chain and isinstance(chain[0], list):
            flattened_steps = []
            for step_list in chain:
                if len(step_list) == 1:
                    flattened_steps.append(step_list[0])
                else:
                    flattened_steps.append(f"[{', '.join(step_list)}]")
            rendered_chain = [_render_step(step) for step in flattened_steps]
        else:
            rendered_chain = [_render_step(step) for step in chain]

        chain_str = " → ".join(rendered_chain)
        formatted.append(f"{idx}. {chain_str}")
    return "; ".join(formatted)


from typing import Any, Dict, List, Optional

def generate_persuader_utterance(
    dialog_history: List[Dict[str, Any]],
    strategy_chain_hint: Optional[str] = None,
    guidance_text: Optional[str] = None,
    model: Optional[str] = None,
    prompt_type: str = "default"
) -> str:
    """
    调用模型生成 Persuader 下一句（ablation-aware）。

    depth  = 是否使用 strategy_chain_hint
    breadth = 是否使用 guidance_text
    """
    dialogue_text = history_to_plain_text(dialog_history)

    use_depth = bool(strategy_chain_hint and str(strategy_chain_hint).strip())
    use_breadth = bool(guidance_text and str(guidance_text).strip())

    if prompt_type in {"Ours", "Ours_1"}:
        from prompts.p4g.runtime_template import build_persuader_generation_prompt

        system_prompt, user_prompt = build_persuader_generation_prompt(
            dialogue_text=dialogue_text,
            strategy_chain_hint=strategy_chain_hint,
            guidance_text=guidance_text,
            prompt_type=prompt_type,
            use_depth=use_depth,         # ✅ NEW
            use_breadth=use_breadth,     # ✅ NEW
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    else:
        from prompts.p4g.runtime_template import build_persuader_generation_prompt

        system_prompt, user_prompt = build_persuader_generation_prompt(
            dialogue_text=dialogue_text,
            strategy_chain_hint=strategy_chain_hint,
            guidance_text=guidance_text,
            prompt_type=prompt_type,
            use_depth=use_depth,
            use_breadth=use_breadth,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # print("[persuader messages]:", messages)
    return chat_completion_persuader(messages, model=model, temperature=0.3)



def generate_persuadee_utterance(
    dialog_history: List[Dict[str, Any]], 
    prompt_type: str = "default",
    model: str = None,
    persona_description: str = None
) -> str:
    """
    生成 Persuadee 下一句。

    persona_description:
        - 若 prompt_type == "personas"，则必须传入
        - 否则忽略
    """

    dialogue_text = history_to_plain_text(dialog_history)

    if prompt_type == "LDPP":
        from prompts.p4g.LDPP import build_persuadee_generation_prompt_ldpp
        system_prompt, user_prompt = build_persuadee_generation_prompt_ldpp(dialogue_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    elif prompt_type == "personas":
        from prompts.p4g.persuadee.persona import build_persuadee_generation_prompt_persona
        system_prompt, user_prompt = build_persuadee_generation_prompt_persona(
            dialogue_text,
            persona_description
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    elif prompt_type == "resistance":
        if persona_description is None:
            raise ValueError("persona_description must be provided when prompt_type is 'resistance'")
        from prompts.p4g.persuadee.resistance import build_persuadee_generation_prompt_resistance
        system_prompt, user_prompt = build_persuadee_generation_prompt_resistance(
            dialogue_text,
            persona_description=persona_description
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    else:
        from prompts.p4g.LDPP import build_persuadee_generation_prompt_ldpp
        system_prompt, user_prompt = build_persuadee_generation_prompt_ldpp(dialogue_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # print("[persuadee_messages]: ", messages)

    return chat_completion_persuadee(messages, model=model,temperature=0.3)
