import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from src.utils.llm_api import (
    call_plato_api,
    call_siliconflow_api,
    # call_openai_api,
)

# P4G critic 的奖励映射和阈值，参考论文：
# [reject, neutral, positive, donate] -> [-1, -0.5, 0.1, 1.0]
ATTITUDE_REWARD = {
    "A": -1.0,   # reject
    "B": -0.5,   # neutral
    "C": 0.1,    # positive
    "D": 1.0,    # donate
}
ETA = 0.6          # 对最终 reward 的阈值：> ETA 视为成功
NUM_SAMPLES = 5    # 每次评估多采样次数（5 个不同接口各评估一次）


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    """
    解析 LLM 返回的文本为 JSON。
    会处理 ```json ... ``` 包裹的情况。
    """
    text = raw_text.strip()

    # 处理 ```json ... ``` 或 ``` ... ``` 的情况
    if text.startswith("```"):
        # 去掉开头的 ```json 或 ```
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # 去掉结尾的 ```
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        data = json.loads(text)
        return data
    except Exception:
        # 解析失败就返回空 dict，外面会用 heuristic 兜底
        return {}


def _build_critic_messages(dialogue_text: str, prompt_type: str):
    """根据 prompt_type 构造 critic 所需的 messages。当前使用 LDPP 风格的 critic 提示。"""
    from prompts.p4g.critic.LDPP import build_critic_prompt_ldpp

    system_prompt, user_prompt = build_critic_prompt_ldpp(dialogue_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def _call_critic_once_with_retry(dialogue_text: str, prompt_type: str, caller, 
                                  max_retries: int = 3, initial_delay: float = 1.0, 
                                  backoff_factor: float = 2.0) -> str:
    """
    带重试机制的 critic 调用，返回单个 attitude（A/B/C/D）。
    
    Args:
        dialogue_text: 对话文本
        prompt_type: prompt 类型
        caller: API 调用函数
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间（秒）
        backoff_factor: 退避因子
    
    Returns:
        态度字符串 (A/B/C/D)
    
    Raises:
        最后一次重试的异常
    """
    messages = _build_critic_messages(dialogue_text, prompt_type)
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # 使用传入的调用函数（四个不同兼容接口之一）
            raw_response = caller(
                messages,
                temperature=1,
                max_tokens=10,
            )

            # 1. 尝试按 JSON 解析
            data = _parse_json_response(raw_response)
            attitude = data.get("attitude")

            # 2. 如果 LDPP prompt 或 JSON 解析失败，使用文本解析
            if attitude not in ["A", "B", "C", "D"]:
                text = raw_response.strip().upper()
                if text.startswith("A") or "EXPLICITLY REFUSED" in text or "REFUSED" in text:
                    attitude = "A"
                elif text.startswith("B") or "REMAINS NEUTRAL" in text or "NEUTRAL" in text:
                    attitude = "B"
                elif text.startswith("C") or "POSITIVE ATTITUDE" in text:
                    attitude = "C"
                elif text.startswith("D") or "DECIDED TO DONATE" in text or "DONATE" in text:
                    attitude = "D"
                else:
                    attitude = None  # ✅ 无法确定：返回 None，让上层跳过

            return attitude
            
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                # 计算延迟时间（指数退避）
                delay = initial_delay * (backoff_factor ** attempt)
                time.sleep(delay)
            else:
                # 最后一次重试也失败了
                raise
    
    # 理论上不会到达这里
    raise last_exception


def _call_critic_once(dialogue_text: str, prompt_type: str, caller) -> str:
    """
    调用一次 critic，返回单个 attitude（A/B/C/D），不做多采样。
    内部使用带重试的版本。
    """
    return _call_critic_once_with_retry(dialogue_text, prompt_type, caller)


from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

def call_critic_model(
    dialog_history: List[Dict[str, Any]],
    prompt_type: str = "default",
    num_samples: int = NUM_SAMPLES,
    max_retries: int = 3,
    return_details: bool = True,
) -> Tuple[str, bool, float, List[str]]:
    """
    返回固定 4 元组：
    - final_attitude: 众数 A/B/C/D
    - should_end: avg_reward 是否超过阈值等
    - avg_reward: mean(ATTITUDE_REWARD[att])
    - sampled_attitudes: 若 return_details=True 返回有效采样列表，否则返回 []
    """

    dialogue_text = "\n".join(
        f"{t.get('speaker','Unknown')}: {t.get('text','')}"
        for t in dialog_history[-4:]
    ) + "\n"

    valid_attitudes: List[str] = []
    valid_rewards: List[float] = []

    try:
        all_callers = [
            call_plato_api,
            call_plato_api,
            call_plato_api,
            call_plato_api,
            call_plato_api,
        ]
        selected_callers = all_callers[: max(1, num_samples)]

        with ThreadPoolExecutor(max_workers=len(selected_callers)) as executor:
            future_to_caller = {
                executor.submit(
                    _call_critic_once_with_retry,
                    dialogue_text,
                    prompt_type,
                    caller,
                    max_retries
                ): caller
                for caller in selected_callers
            }

            for future in as_completed(future_to_caller):
                caller = future_to_caller[future]
                try:
                    att = future.result()
                except Exception as e:
                    print(f"[Critic] {getattr(caller, '__name__', 'caller')} failed after {max_retries} retries: {e}")
                    continue

                if att in ("A", "B", "C", "D"):
                    valid_attitudes.append(att)
                    valid_rewards.append(ATTITUDE_REWARD[att])

        if not valid_attitudes:
            # 安全默认：中立
            final_attitude = "B"
            avg_reward = ATTITUDE_REWARD["B"]
            should_end = False  # 这里你也可以按你的策略设定
            sampled = valid_attitudes if return_details else []
            return final_attitude, should_end, avg_reward, sampled

        final_attitude = Counter(valid_attitudes).most_common(1)[0][0]
        avg_reward = sum(valid_rewards) / len(valid_rewards)

        print(f"[avg_reward]: {avg_reward} (valid={len(valid_rewards)}/{len(selected_callers)})")
        print("================================================")

        should_end = avg_reward > ETA
        sampled = valid_attitudes if return_details else []
        return final_attitude, should_end, avg_reward, sampled

    except Exception as e:
        print(f"Critic model call failed: {e}")
        # 安全默认：中立 & 不结束 & 中立 reward
        return "B", False, ATTITUDE_REWARD["B"], ([] if not return_details else [])
