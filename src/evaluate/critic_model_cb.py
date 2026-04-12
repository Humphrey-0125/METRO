import json
import time
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
from typing import List, Dict, Any, Tuple, Optional

from src.utils.llm_api import call_llm_api

# ==============
# Config
# ==============
NUM_SAMPLES = int(5)
REQUEST_MAX_TOKENS = 30
TEMPERATURE = 0.0

MAX_RETRIES_DEFAULT = 3
INITIAL_DELAY = 1.0
BACKOFF_FACTOR = 2.0


# ==============
# JSON parse helpers
# ==============
def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    t = _strip_code_fence(raw_text)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _normalize_deal_price(data: Dict[str, Any]) -> Tuple[Optional[bool], Optional[float]]:
    if not isinstance(data, dict):
        return None, None

    deal = data.get("deal", None)
    price = data.get("price", None)

    # normalize deal
    if isinstance(deal, bool):
        pass
    elif isinstance(deal, str):
        dl = deal.strip().lower()
        if dl in {"true", "yes", "y", "1"}:
            deal = True
        elif dl in {"false", "no", "n", "0"}:
            deal = False
        else:
            deal = None
    else:
        deal = None

    # normalize price
    if deal is False:
        return False, None

    if deal is True:
        if price is None:
            return True, None
        if isinstance(price, (int, float)):
            return True, float(price)

        if isinstance(price, str):
            s = price.strip()
            m = re.search(r"(-?\d+(?:\.\d+)?)", s)
            if not m:
                return True, None
            try:
                return True, float(m.group(1))
            except Exception:
                return True, None

        return True, None

    return None, None


# ==============
# Prompt builder
# ==============
def _build_critic_messages(dialogue_text: str, prompt_type: str = "DPDP"):
    if prompt_type != "DPDP":
        prompt_type = "DPDP"

    from prompts.cb.critic.DPDP import build_cb_judge_prompt

    system_prompt, user_prompt = build_cb_judge_prompt(dialogue_text)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ==============
# Call once with retry
# ==============
def _call_critic_once_with_retry(
    dialogue_text: str,
    prompt_type: str,
    caller,
    max_retries: int = MAX_RETRIES_DEFAULT,
    initial_delay: float = INITIAL_DELAY,
    backoff_factor: float = BACKOFF_FACTOR,
) -> Tuple[Optional[bool], Optional[float], str]:
    messages = _build_critic_messages(dialogue_text, prompt_type)
    # print(f"messages: {messages}")

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            raw = caller(
                messages,
                temperature=TEMPERATURE,
                max_tokens=REQUEST_MAX_TOKENS,
            )
            data = _parse_json_response(raw)
            print(f"data: {data}")
            deal, price = _normalize_deal_price(data)

            if deal is not None:
                return deal, price, raw

            last_exception = ValueError(f"Invalid critic output: {raw[:200]}")
            raise last_exception

        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = initial_delay * (backoff_factor ** attempt)
                time.sleep(delay)
            else:
                return None, None, str(e)

    return None, None, str(last_exception)


# ==============
# Voting aggregation
# ==============
def _aggregate_votes(votes: List[Tuple[Optional[bool], Optional[float], str]]) -> Tuple[bool, Optional[float]]:
    deals = [v[0] for v in votes if v[0] is not None]
    if not deals:
        return False, None

    final_deal = all(deal is True for deal in deals)

    if not final_deal:
        return False, None

    prices = []
    for deal, price, _raw in votes:
        if deal is True and isinstance(price, (int, float)):
            prices.append(float(price))

    if not prices:
        return True, None

    pc = Counter(prices)
    most_common = pc.most_common()
    if len(most_common) == 1:
        return True, most_common[0][0]

    best_count = most_common[0][1]
    tied = [p for p, cnt in most_common if cnt == best_count]
    if len(tied) == 1:
        return True, tied[0]

    return True, float(median(prices))


# ==============
# Public API
# ==============
def call_critic_model(
    dialog_history: List[Dict[str, Any]],
    prompt_type: str = "DPDP",
    num_samples: int = NUM_SAMPLES,
    max_retries: int = MAX_RETRIES_DEFAULT,
    return_details: bool = True,
) -> Tuple[bool, bool, Optional[float], List[Dict[str, Any]]]:

    filtered_turns = [t for t in dialog_history if t.get("text") is not None]
    last_four_turns = filtered_turns[-4:] if len(filtered_turns) > 4 else filtered_turns
    dialogue_text = "\n".join(
        f"{t.get('speaker','Unknown')}: {t.get('text','')}"
        for t in last_four_turns
    ).strip()
    # dialogue_text = "\n".join(
    #     f"{t.get('speaker','Unknown')}: {t.get('text','')}"
    #     for t in dialog_history
    # ).strip()    

    all_callers = [
        call_llm_api,
        call_llm_api,
        call_llm_api,
        call_llm_api,
        call_llm_api,
    ]
    selected_callers = all_callers[: max(len(all_callers), 1)]

    votes: List[Tuple[Optional[bool], Optional[float], str]] = []
    details: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=len(selected_callers)) as ex:
        futures = [
            ex.submit(
                _call_critic_once_with_retry,
                dialogue_text,
                prompt_type,
                caller,
                max_retries,
            )
            for caller in selected_callers
        ]

        for fut in as_completed(futures):
            deal, price, raw = fut.result()
            votes.append((deal, price, raw))
            if return_details:
                details.append(
                    {
                        "deal": deal,
                        "price": price,
                        "raw": raw[:500],
                    }
                )

    final_deal, final_price = _aggregate_votes(votes)
    should_end = bool(final_deal)

    if not final_deal:
        final_price = None

    return final_deal, should_end, final_price, (details if return_details else [])
