"""
Unicode-based language detection for .lab file transcripts.

Detects language by analyzing Unicode character ranges in text.
Returns GPT-SoVITS compatible language codes: KO, JA, ZH, EN, etc.
"""

import re
from collections import Counter


# Unicode ranges for each language
UNICODE_RANGES = {
    'KO': [
        (0xAC00, 0xD7AF),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0x3130, 0x318F),  # Hangul Compatibility Jamo
        (0xA960, 0xA97F),  # Hangul Jamo Extended-A
        (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    ],
    'JA': [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
        (0xFF65, 0xFF9F),  # Halfwidth Katakana
    ],
    'ZH': [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    ],
    'TH': [
        (0x0E00, 0x0E7F),  # Thai
    ],
    'AR': [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
    ],
    'RU': [
        (0x0400, 0x04FF),  # Cyrillic
        (0x0500, 0x052F),  # Cyrillic Supplement
    ],
}


def _get_char_language(char: str) -> str | None:
    """Determine which language a character belongs to."""
    code_point = ord(char)

    for lang, ranges in UNICODE_RANGES.items():
        for start, end in ranges:
            if start <= code_point <= end:
                return lang

    return None


def _is_ascii_letter(char: str) -> bool:
    """Check if character is ASCII letter (A-Z, a-z)."""
    return char.isascii() and char.isalpha()


def detect_language(text: str) -> str:
    """
    Detect language from text using Unicode character analysis.

    Priority:
    1. If Korean characters exist -> KO
    2. If Japanese-specific characters (hiragana/katakana) exist -> JA
    3. If Chinese characters exist (and no Japanese) -> ZH
    4. Otherwise -> EN (default)

    Args:
        text: Input text to analyze

    Returns:
        Language code: 'KO', 'JA', 'ZH', 'EN', etc.
    """
    if not text or not text.strip():
        return 'EN'

    # Count characters by language
    lang_counts: Counter[str] = Counter()
    ascii_count = 0
    total_letters = 0

    for char in text:
        if char.isspace() or char in '.,!?;:\'"()-[]{}':
            continue

        lang = _get_char_language(char)
        if lang:
            lang_counts[lang] += 1
            total_letters += 1
        elif _is_ascii_letter(char):
            ascii_count += 1
            total_letters += 1

    if total_letters == 0:
        return 'EN'

    # Priority-based detection
    # 1. Korean has highest priority (clear distinction)
    if lang_counts.get('KO', 0) > 0:
        return 'KO'

    # 2. Japanese (hiragana/katakana are unique to Japanese)
    if lang_counts.get('JA', 0) > 0:
        return 'JA'

    # 3. Chinese (CJK ideographs without Japanese kana)
    if lang_counts.get('ZH', 0) > 0:
        return 'ZH'

    # 4. Other specific languages
    if lang_counts.get('TH', 0) > 0:
        return 'TH'
    if lang_counts.get('AR', 0) > 0:
        return 'AR'
    if lang_counts.get('RU', 0) > 0:
        return 'RU'

    # 5. Default to English
    return 'EN'


def read_lab_file(lab_path: str) -> str:
    """
    Read text content from a .lab file.

    .lab files are simple text files containing transcription.
    Supports various encodings (UTF-8, UTF-16, etc.)

    Args:
        lab_path: Path to .lab file

    Returns:
        Text content from the file
    """
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp949', 'shift_jis', 'gb2312']

    for encoding in encodings:
        try:
            with open(lab_path, 'r', encoding=encoding) as f:
                content = f.read().strip()
                # Remove BOM if present
                if content.startswith('\ufeff'):
                    content = content[1:]
                return content
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Fallback: read as binary and decode with errors='replace'
    with open(lab_path, 'rb') as f:
        return f.read().decode('utf-8', errors='replace').strip()


if __name__ == '__main__':
    # Test cases
    test_cases = [
        ("안녕하세요, 반갑습니다!", "KO"),
        ("こんにちは、元気ですか？", "JA"),
        ("你好，世界！", "ZH"),
        ("Hello, world!", "EN"),
        ("これは日本語と English の混合です", "JA"),
        ("한국어와 English 혼합 텍스트", "KO"),
        ("中文和English混合", "ZH"),
        ("비-케어뽈! 트랩이 보여.", "KO"),
    ]

    print("Language Detection Test:")
    print("-" * 50)
    for text, expected in test_cases:
        detected = detect_language(text)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{text[:30]}...' -> {detected} (expected: {expected})")
