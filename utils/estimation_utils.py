import re
import logging
from config import settings

def estimate_script_length_minutes(script_text: str) -> float:
    """Estimates the spoken length of the script text in minutes."""
    if not script_text or not script_text.strip(): return 0

    word_count = len(re.findall(r'\w+', script_text))
    estimated_minutes = word_count / settings.WORDS_PER_MINUTE_NARRATION

    logging.info(f"Estimated script length: {word_count} words, approx. {estimated_minutes:.2f} minutes.")
    return estimated_minutes