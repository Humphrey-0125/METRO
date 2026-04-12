import random
import json
from typing import List, Dict, Any

def load_personas(path="outputs/P4G/personas/personas_eval.jsonl") -> Dict[str, str]:
    personas = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["pid"]
            desc = obj["description"]
            personas[pid] = desc
    return personas


def load_dev_dataset() -> List[Dict[str, Any]]:
    dev_file = "data/P4G/dev.json"
    try:
        with open(dev_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {dev_file}")
        return []