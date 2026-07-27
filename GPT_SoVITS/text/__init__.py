"""Map cleaned phoneme strings to symbol-ID sequences per model version."""

import os

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

_symbol_to_id_v1 = {s: i for i, s in enumerate(symbols_v1.symbols)}
_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}


def cleaned_text_to_sequence(cleaned_text, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        phones = [_symbol_to_id_v1[symbol] for symbol in cleaned_text]
    else:
        phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]

    return phones
