"""
텍스트 전처리 유틸리티 (konlpy 불필요)
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """
    텍스트 정제 (간단 버전)

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    if not isinstance(text, str):
        return ""

    # 1. 소문자 변환
    text = text.lower()

    # 2. 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
    text = re.sub(r"[^가-힣a-z0-9\s]", " ", text)

    # 3. 연속된 공백을 하나로
    text = re.sub(r"\s+", " ", text)

    # 4. 앞뒤 공백 제거
    text = text.strip()

    return text


def tokenize(text: str) -> list:
    """
    간단한 토큰화 (공백 기준)

    Args:
        text: 텍스트

    Returns:
        토큰 리스트
    """
    if not isinstance(text, str):
        return []

    # 정제 후 공백으로 분리
    cleaned = clean_text(text)
    tokens = cleaned.split()

    return tokens


def remove_stopwords(tokens: list) -> list:
    """
    불용어 제거 (한국어 기본 불용어)

    Args:
        tokens: 토큰 리스트

    Returns:
        불용어가 제거된 토큰 리스트
    """
    # 한국어 기본 불용어 (조사, 어미 등)
    stopwords = {
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "의",
        "와",
        "과",
        "도",
        "으로",
        "로",
        "에서",
        "까지",
        "부터",
        "한",
        "하다",
        "있다",
        "없다",
        "되다",
        "이다",
        "아니다",
        "것",
        "수",
        "등",
    }

    # 1글자 토큰과 불용어 제거
    filtered = [token for token in tokens if len(token) > 1 and token not in stopwords]

    return filtered


def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    키워드 추출 (빈도 기반)

    Args:
        text: 텍스트
        top_n: 추출할 키워드 개수

    Returns:
        키워드 리스트
    """
    from collections import Counter

    # 토큰화 및 불용어 제거
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens)

    # 빈도 계산
    counter = Counter(filtered)
    keywords = [word for word, count in counter.most_common(top_n)]

    return keywords


def negative_keyword_ratio(texts, negative_keywords):
    if len(texts) == 0:
        return 0.0

    count = 0
    for t in texts:
        if any(k in t for k in negative_keywords):
            count += 1
    return count / len(texts)


def duplicate_ratio(texts, threshold=0.8):
    # 빈 문자열 제거
    texts = [t for t in texts if isinstance(t, str) and t.strip() != ""]

    # 유효 문장이 2개 미만이면 중복 불가
    if len(texts) < 2:
        return 0.0

    try:
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(texts)
        sim = cosine_similarity(X)

        dup_count = 0
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i][j] > threshold:
                    dup_count += 1

        return dup_count / (n + 1)

    except ValueError:
        # empty vocabulary 같은 에러 방어
        return 0.0


if __name__ == "__main__":
    # 테스트
    test_text = "이 제품은 정말 좋아요! 강력 추천합니다. 배송도 빠르고 품질도 최고예요."

    print("=== 텍스트 전처리 테스트 ===")
    print(f"원본: {test_text}")
    print(f"정제: {clean_text(test_text)}")
    print(f"토큰: {tokenize(test_text)}")
    print(f"불용어 제거: {remove_stopwords(tokenize(test_text))}")
    print(f"키워드: {extract_keywords(test_text)}")
