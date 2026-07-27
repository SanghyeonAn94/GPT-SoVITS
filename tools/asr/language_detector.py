"""Unicode-based language detection for .lab file transcripts, returning GPT-SoVITS language codes (KO, JA, ZH, EN, ...) by analyzing Unicode character ranges; also reads .lab files with multi-encoding fallback."""

import re
from collections import Counter


UNICODE_RANGES = {
    'KO': [
        (0xAC00, 0xD7AF),
        (0x1100, 0x11FF),
        (0x3130, 0x318F),
        (0xA960, 0xA97F),
        (0xD7B0, 0xD7FF),
    ],
    'JA': [
        (0x3040, 0x309F),
        (0x30A0, 0x30FF),
        (0x31F0, 0x31FF),
        (0xFF65, 0xFF9F),
    ],
    'ZH': [
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0xF900, 0xFAFF),
    ],
    'TH': [
        (0x0E00, 0x0E7F),
    ],
    'AR': [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
    ],
    'RU': [
        (0x0400, 0x04FF),
        (0x0500, 0x052F),
    ],
}


def _get_char_language(char: str) -> str | None:
    code_point = ord(char)

    for lang, ranges in UNICODE_RANGES.items():
        for start, end in ranges:
            if start <= code_point <= end:
                return lang

    return None


def _is_ascii_letter(char: str) -> bool:
    return char.isascii() and char.isalpha()


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return 'EN'

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

    if lang_counts.get('KO', 0) > 0:
        return 'KO'

    if lang_counts.get('JA', 0) > 0:
        return 'JA'

    if lang_counts.get('ZH', 0) > 0:
        return 'ZH'

    if lang_counts.get('TH', 0) > 0:
        return 'TH'
    if lang_counts.get('AR', 0) > 0:
        return 'AR'
    if lang_counts.get('RU', 0) > 0:
        return 'RU'

    return 'EN'


def read_lab_file(lab_path: str) -> str:
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp949', 'shift_jis', 'gb2312']

    for encoding in encodings:
        try:
            with open(lab_path, 'r', encoding=encoding) as f:
                content = f.read().strip()
                if content.startswith('﻿'):
                    content = content[1:]
                return content
        except (UnicodeDecodeError, UnicodeError):
            continue

    with open(lab_path, 'rb') as f:
        return f.read().decode('utf-8', errors='replace').strip()


if __name__ == '__main__':
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
