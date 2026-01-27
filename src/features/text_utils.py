import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

okt = Okt()

# ------------------
# 기본 전처리
# ------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return okt.morphs(text)


# ------------------
# 리뷰 길이
# ------------------
def text_length(text: str) -> int:
    return len(text)


# ------------------
# 부정 키워드 비율
# ------------------
def negative_keyword_ratio(texts, negative_keywords):
    if len(texts) == 0:
        return 0.0

    count = 0
    for t in texts:
        if any(k in t for k in negative_keywords):
            count += 1
    return count / len(texts)


# ------------------
# 중복 리뷰 비율
# ------------------
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

