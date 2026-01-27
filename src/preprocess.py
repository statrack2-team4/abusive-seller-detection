import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^가-힣a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
