import json
import time
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional

from src.utils.llm_api import call_llm_api

ATTITUDE_REWARD = {
    "A": -1.0,
    "B": -0.5,
    "C": 0.1,
    "D": 1.0,
}
ETA = 0.6
NUM_SAMPLES = 5


def _normalize_dialogue_text(dialog_history: List[Dict[str, Any]], last_k: int = 8) -> str:
    turns = dialog_history[-last_k:] if last_k and last_k > 0 else dialog_history
    return "\n".join(f"{t.get('speaker','Unknown')}: {t.get('text','')}" for t in turns) + "\n"


def _build_critic_messages(emotion_type: str, problem_type: str, dialogue_text: str):
    """
    构造 ESConv critic messages（不含 scene）
    """
    from prompts.esc.critic.standard import build_esc_judge_prompt
    # build_esc_judge_prompt(emotion_type, problem_type, dialogue_text)

    system_prompt, user_prompt = build_esc_judge_prompt(emotion_type, problem_type, dialogue_text)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_attitude(raw_response: str) -> Optional[str]:
    if not raw_response:
        return None

    text = raw_response.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()

    upper = text.strip().upper()

    if upper and upper[0] in ("A", "B", "C", "D"):
        return upper[0]

    m = re.search(r"\b([ABCD])\b", upper)
    if m:
        return m.group(1)

    return None


def _call_critic_once_with_retry(
    emotion_type: str,
    problem_type: str,
    dialogue_text: str,
    caller,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Optional[str]:
    """
    带重试的 critic 调用，返回 attitude (A/B/C/D) 或 None
    """
    messages = _build_critic_messages(emotion_type, problem_type, dialogue_text)
    # print("critic message:",messages)
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            raw_response = caller(
                messages,
                temperature=1,
                max_tokens=10,
            )
            return _extract_attitude(raw_response)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = initial_delay * (backoff_factor ** attempt)
                time.sleep(delay)
            else:
                raise

    raise last_exception


def call_critic_model(
    dialog_history: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    num_samples: int = NUM_SAMPLES,
    max_retries: int = 3,
    return_details: bool = True,
) -> Tuple[str, bool, float, List[str]]:

    meta = meta or {}
    emotion_type = (meta.get("emotion_type") or "").strip() or "N/A"
    problem_type = (meta.get("problem_type") or "").strip() or "N/A"

    dialogue_text = _normalize_dialogue_text(dialog_history, last_k=10)

    valid_attitudes: List[str] = []
    valid_rewards: List[float] = []

    try:
        all_callers = [
            call_llm_api,
            call_llm_api,
            call_llm_api,
            call_llm_api,
            call_llm_api,
        ]
        selected_callers = all_callers[: max(1, num_samples)]

        with ThreadPoolExecutor(max_workers=len(selected_callers)) as executor:
            future_to_caller = {
                executor.submit(
                    _call_critic_once_with_retry,
                    emotion_type,
                    problem_type,
                    dialogue_text,
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
            final_attitude = "B"
            avg_reward = ATTITUDE_REWARD["B"]
            should_end = False
            sampled = valid_attitudes if return_details else []
            return final_attitude, should_end, avg_reward, sampled

        final_attitude = Counter(valid_attitudes).most_common(1)[0][0]
        avg_reward = sum(valid_rewards) / len(valid_rewards)

        print(f"[avg_reward]: {avg_reward} (valid={len(valid_rewards)}/{len(selected_callers)})")
        print("================================================")

        should_end = avg_reward > ETA or avg_reward < -0.7
        sampled = valid_attitudes if return_details else []
        return final_attitude, should_end, avg_reward, sampled

    except Exception as e:
        print(f"Critic model call failed: {e}")
        return "B", False, ATTITUDE_REWARD["B"], ([] if not return_details else [])