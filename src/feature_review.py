"""
리뷰 Feature 생성 모듈
"""

import pandas as pd
import numpy as np
from src.features.text_utils import clean_text

# 중복 리뷰 비율 계산을 위한 추가 import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 부정 키워드 사전
NEGATIVE_KEYWORDS = [
    "불량", "고장", "환불", "가짜", "사기", "효과없음",
    "부작용", "작동안됨", "충전안됨", "트러블", "피부염",
    "반품", "취소", "하자", "파손", "먹튀"
]


def text_length(text: str) -> int:
    """텍스트 길이 계산"""
    if not isinstance(text, str):
        return 0
    return len(text)


def negative_keyword_ratio(texts, negative_keywords):
    """부정 키워드 포함 비율 계산"""
    if len(texts) == 0:
        return 0.0
    
    count = 0
    for t in texts:
        if any(k in t for k in negative_keywords):
            count += 1
    return count / len(texts)


def duplicate_ratio(texts, threshold=0.8):
    """중복 리뷰 비율 계산 (TF-IDF 유사도 기반)"""
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


def add_review_features(reviews: pd.DataFrame):
    """
    리뷰 개별 row에 기본 feature 추가
    
    Args:
        reviews: 리뷰 데이터프레임 (review_text, review_rating 포함)
    
    Returns:
        feature가 추가된 데이터프레임
    """
    reviews = reviews.copy()
    
    # 텍스트 전처리
    reviews["clean_text"] = reviews["review_text"].fillna("").apply(clean_text)
    
    # 텍스트 길이
    reviews["text_length"] = reviews["clean_text"].apply(text_length)
    
    # 텍스트 없는 리뷰 (5자 미만)
    reviews["is_textless"] = (reviews["text_length"] < 5)
    
    # 5점 리뷰 여부
    reviews["is_5star"] = (reviews["review_rating"] == 5)
    
    # 저평점 리뷰 여부 (1~2점)
    reviews["is_low_rating"] = (reviews["review_rating"] <= 2)
    
    # 부정 키워드 포함 여부
    reviews["has_negative_keyword"] = reviews["clean_text"].apply(
        lambda x: any(kw in x for kw in NEGATIVE_KEYWORDS)
    )
    
    return reviews


def aggregate_review_by_seller(reviews: pd.DataFrame, products: pd.DataFrame):
    """
    판매자 단위로 리뷰 feature 집계
    
    문서의 7.1 리뷰 Feature 전체 구현:
    - 평점: 평균 평점, 평점 분산, 저평점 비율
    - 텍스트: 평균 리뷰 길이, 중복 리뷰 비율
    - 키워드: 부정 키워드 비율
    
    Args:
        reviews: add_review_features가 적용된 리뷰 데이터프레임
        products: 상품 데이터프레임 (product_id, vendor_name 포함)
    
    Returns:
        판매자 단위 집계 데이터프레임
    """
    # product_id → vendor_name 매핑
    prod_vendor = products[["product_id", "vendor_name"]].drop_duplicates()
    merged = reviews.merge(prod_vendor, on="product_id", how="left")
    
    # 판매자별 집계
    result = []
    
    for vendor, g in merged.groupby("vendor_name"):
        texts = g["clean_text"].tolist()
        ratings = g["review_rating"]
        
        # 평점 feature
        avg_rating = ratings.mean()
        rating_std = ratings.std() if len(ratings) > 1 else 0
        low_rating_ratio = g["is_low_rating"].mean()
        
        # 텍스트 feature
        avg_review_length = g["text_length"].mean()
        duplicate_review_ratio = duplicate_ratio(texts, threshold=0.8)
        
        # 키워드 feature
        negative_keyword_ratio_val = negative_keyword_ratio(texts, NEGATIVE_KEYWORDS)
        
        # 텍스트 없는 5점 리뷰 비율 (조작 의심)
        textless_5star = g[g["is_5star"]]["is_textless"].sum()
        total_5star = g["is_5star"].sum()
        textless_5star_ratio = textless_5star / total_5star if total_5star > 0 else 0
        
        feature = {
            "vendor_name": vendor,
            # 평점 feature
            "avg_rating": avg_rating,
            "rating_std": rating_std,
            "low_rating_ratio": low_rating_ratio,
            # 텍스트 feature
            "avg_review_length": avg_review_length,
            "duplicate_review_ratio": duplicate_review_ratio,
            # 키워드 feature
            "negative_keyword_ratio": negative_keyword_ratio_val,
            # 추가 feature
            "textless_5star_ratio": textless_5star_ratio,
            "review_count": len(g)
        }
        
        result.append(feature)
    
    return pd.DataFrame(result)


def calculate_review_density(seller_df: pd.DataFrame, products: pd.DataFrame):
    """
    핵심 가설 검증 변수: 리뷰_밀도 = 총_리뷰_수 / 등록_상품_수
    
    Args:
        seller_df: aggregate_review_by_seller 결과 (review_count 포함)
        products: 상품 데이터프레임 (vendor_name 포함)
    
    Returns:
        review_density 컬럼이 추가된 데이터프레임
    """
    seller_df = seller_df.copy()
    
    # 판매자별 상품 수 계산
    product_counts = products.groupby("vendor_name").size().reset_index(name="product_count")
    
    # 병합
    seller_df = seller_df.merge(product_counts, on="vendor_name", how="left")
    
    # 리뷰 밀도 계산 (상품 수가 0이면 0으로 처리)
    seller_df["product_count"] = seller_df["product_count"].fillna(0)
    seller_df["review_density"] = np.where(
        seller_df["product_count"] > 0,
        seller_df["review_count"] / seller_df["product_count"],
        0
    )
    
    return seller_df


def add_sentiment_features(seller_df: pd.DataFrame, sentiment_results: pd.DataFrame):
    """
    감성 분석 결과를 판매자 데이터에 병합
    
    이 함수는 별도의 감성 분석 모듈에서 생성된 결과를 받아 추가
    
    Args:
        seller_df: 판매자 집계 데이터
        sentiment_results: 판매자별 감성 분석 결과
            필수 컬럼: vendor_name, negative_sentiment_ratio, avg_sentiment_score
    
    Returns:
        감성 feature가 추가된 데이터프레임
    """
    seller_df = seller_df.copy()
    
    # 감성 결과 병합
    seller_df = seller_df.merge(
        sentiment_results[["vendor_name", "negative_sentiment_ratio", "avg_sentiment_score"]],
        on="vendor_name",
        how="left"
    )
    
    # 누락값 처리 (감성 분석 결과 없는 경우 0으로)
    seller_df["negative_sentiment_ratio"] = seller_df["negative_sentiment_ratio"].fillna(0)
    seller_df["avg_sentiment_score"] = seller_df["avg_sentiment_score"].fillna(0.5)  # 중립
    
    return seller_df


def calculate_rating_sentiment_gap(seller_df: pd.DataFrame):
    """
    평점-감성 괴리도 계산: |평점 정규화값 - 감성점수|
    
    평점 5점 만점 → 0~1로 정규화 후 감성 점수(0~1)와 차이 계산
    
    Args:
        seller_df: avg_rating, avg_sentiment_score 포함
    
    Returns:
        rating_sentiment_gap 컬럼이 추가된 데이터프레임
    """
    seller_df = seller_df.copy()
    
    # 평점 정규화 (1~5 → 0~1)
    seller_df["rating_normalized"] = (seller_df["avg_rating"] - 1) / 4
    
    # 괴리도 계산
    seller_df["rating_sentiment_gap"] = np.abs(
        seller_df["rating_normalized"] - seller_df["avg_sentiment_score"]
    )
    
    return seller_df