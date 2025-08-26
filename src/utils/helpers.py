import hashlib
import re
from typing import Optional

def parse_currency_string(value: str) -> Optional[float]:
    if not value or value is None:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    is_negative = False
    if value_str.startswith("(") and value_str.endswith(")"):
        is_negative = True
        value_str = value_str[1:-1]

    value_str = re.sub(r"[A-Z]{3}", "", value_str)
    value_str = re.sub(r"[$€£¥₹]", "", value_str)
    value_str = value_str.replace(",", "").replace(" ", "")

    try:
        parsed_value = float(value_str)
        return -parsed_value if is_negative else parsed_value
    except (ValueError, TypeError):
        return None
    
def calculate_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
