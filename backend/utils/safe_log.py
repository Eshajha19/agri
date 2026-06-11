import re

CONTROL_CHARS = re.compile(r'[\x00-\x1F\x7F]')  # ASCII control chars
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')  # ANSI escape sequences

def sanitize_log_field(value: str) -> str:
    """
    Remove control characters and ANSI escape sequences from externally sourced strings.
    Ensures logs cannot be injected with terminal codes or fake lines.
    """
    if not isinstance(value, str):
        return str(value)
    value = CONTROL_CHARS.sub('', value)
    value = ANSI_ESCAPE.sub('', value)
    return value
