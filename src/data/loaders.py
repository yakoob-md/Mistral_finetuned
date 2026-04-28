# Data loading utilities for JSONL files
import json

def load_jsonl(file_path):
    """
    Load data from a JSONL file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
