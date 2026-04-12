import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from src.utils.llm_api import call_llm_api

# [reject, neutral, positive, donate] -> [-1, -0.5, 0.1, 1.0]
ATTITUDE_REWARD = {
    "A": -1.0,   # reject
    "B": -0.5,   # neutral
    "C": 0.1,    # positive
    "D": 1.0,    # donate
}
ETA = 0.6
NUM_SAMPLES = 5


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        data = json.loads(text)
        return data
    except Exception:
        return {}


def _build_critic_messages(dialogue_text: str, prompt_type: str):
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
    messages = _build_critic_messages(dialogue_text, prompt_type)
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            raw_response = caller(
                messages,
                temperature=1,
                max_tokens=10,
            )

            data = _parse_json_response(raw_response)
            attitude = data.get("attitude")

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
                    attitude = None

            return attitude
            
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                delay = initial_delay * (backoff_factor ** attempt)
                time.sleep(delay)
            else:
                raise
    
    raise last_exception


def _call_critic_once(dialogue_text: str, prompt_type: str, caller) -> str:
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

    dialogue_text = "\n".join(
        f"{t.get('speaker','Unknown')}: {t.get('text','')}"
        for t in dialog_history[-4:]
    ) + "\n"

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
            final_attitude = "B"
            avg_reward = ATTITUDE_REWARD["B"]
            should_end = False
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
        return "B", False, ATTITUDE_REWARD["B"], ([] if not return_details else [])
