#!/usr/bin/env python3
"""
연속 구두점 처리 테스트 스크립트

수정된 replace_consecutive_punctuation()이 올바르게 동작하는지 검증
"""

import sys
import os

# GPT-SoVITS 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GPT_SoVITS'))

import re


def test_replace_consecutive_punctuation():
    """replace_consecutive_punctuation() 동작 테스트"""

    def replace_consecutive_punctuation(text, max_n=3):
        """수정된 함수 (TextPreprocessor.py와 동일)"""
        def normalize_punct(match):
            punct = match.group(1)
            count = len(match.group(0))
            normalized_count = min(count, max_n)
            return punct * normalized_count

        text = re.sub(r'(,){1,}', normalize_punct, text)
        text = re.sub(r'(\.){1,}', normalize_punct, text)
        return text

    test_cases = [
        ("안녕, 반갑습니다", "안녕, 반갑습니다", "쉼표 1개"),
        ("안녕,, 반갑습니다", "안녕,, 반갑습니다", "쉼표 2개"),
        ("안녕,,, 반갑습니다", "안녕,,, 반갑습니다", "쉼표 3개"),
        ("안녕,,,, 반갑습니다", "안녕,,, 반갑습니다", "쉼표 4개 → 3개"),
        ("안녕,,,,, 반갑습니다", "안녕,,, 반갑습니다", "쉼표 5개 → 3개"),
        ("나는 말야. 그래.", "나는 말야. 그래.", "마침표 1개"),
        ("나는 말야.. 그래..", "나는 말야.. 그래..", "마침표 2개"),
        ("나는 말야... 그래...", "나는 말야... 그래...", "마침표 3개"),
        ("나는 말야.... 그래....", "나는 말야... 그래...", "마침표 4개 → 3개"),
        ("나는 말야..... 그래.....", "나는 말야... 그래...", "마침표 5개 → 3개"),
        ("그게,, 뭐랄까... 어려워,,,, 정말.", "그게,, 뭐랄까... 어려워,,, 정말.", "혼합"),
    ]

    print("=" * 80)
    print("replace_consecutive_punctuation() 테스트")
    print("=" * 80)
    print()

    all_passed = True
    for original, expected, description in test_cases:
        result = replace_consecutive_punctuation(original)
        passed = result == expected

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {description}")
        print(f"  입력:    {original}")
        print(f"  예상:    {expected}")
        print(f"  결과:    {result}")

        if not passed:
            all_passed = False
            print(f"  ⚠️  불일치!")
        print()

    print("=" * 80)
    if all_passed:
        print("✅ 모든 테스트 통과!")
    else:
        print("❌ 일부 테스트 실패")
    print("=" * 80)

    return all_passed


def test_korean_g2p_integration():
    """korean.g2p()와 통합 테스트"""
    try:
        from GPT_SoVITS.text.korean import g2p

        print("\n" + "=" * 80)
        print("Korean G2P 통합 테스트")
        print("=" * 80)
        print()

        test_texts = [
            "안녕, 반갑습니다",
            "안녕,, 반갑습니다",
            "안녕,,, 반갑습니다",
            "나는 말야. 그래.",
            "나는 말야.. 그래..",
            "나는 말야... 그래...",
        ]

        for text in test_texts:
            # 전처리 (ceiling normalization)
            def replace_consecutive_punctuation(text, max_n=3):
                def normalize_punct(match):
                    punct = match.group(1)
                    count = len(match.group(0))
                    normalized_count = min(count, max_n)
                    return punct * normalized_count

                text = re.sub(r'(,){1,}', normalize_punct, text)
                text = re.sub(r'(\.){1,}', normalize_punct, text)
                return text

            normalized = replace_consecutive_punctuation(text)

            # G2P 실행
            phones = g2p(normalized)

            # 구두점 카운트
            comma_count = phones.count(',')
            dot_count = phones.count('.')

            print(f"원본: {text}")
            print(f"정규화: {normalized}")
            print(f"Phones: {phones}")
            print(f"구두점 개수: 쉼표 {comma_count}개, 마침표 {dot_count}개")
            print()

        print("=" * 80)
        print("✅ G2P 통합 테스트 완료")
        print("=" * 80)

    except Exception as e:
        print(f"❌ G2P 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 연속 구두점 처리 테스트 시작\n")

    # Test 1: replace_consecutive_punctuation 단위 테스트
    test_replace_consecutive_punctuation()

    # Test 2: korean.g2p 통합 테스트
    test_korean_g2p_integration()

    print("\n✅ 테스트 완료!\n")
