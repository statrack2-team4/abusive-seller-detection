"""
문의 Feature 생성 모듈
"""

import pandas as pd
import numpy as np
from src.features.text_utils import clean_text


# 문의 유형별 키워드 사전
QUESTION_TYPE_KEYWORDS = {
    "배송": ["언제", "배송", "도착", "출고", "발송"],
    "환불": ["환불", "취소", "반품", "교환"],
    "불량": ["불량", "고장", "파손", "하자", "작동", "안됨"],
    "진위": ["정품", "가품", "정식수입", "병행수입", "진짜"],
    "옵션": ["색상", "사이즈", "용량", "크기", "사양"],
    "사용법": ["어떻게", "사용법", "복용", "방법", "쓰는법"]
}


def classify_question_type(text: str) -> str:
    """
    문의 텍스트를 유형별로 분류 (키워드 기반)
    
    Args:
        text: 문의 텍스트
    
    Returns:
        문의 유형 (배송/환불/불량/진위/옵션/사용법/기타)
    """
    text = clean_text(text)
    
    for qtype, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return qtype
    
    return "기타"


def add_question_features(questions: pd.DataFrame):
    """
    문의 개별 row에 기본 feature 추가
    
    Args:
        questions: 문의 데이터프레임 (question 컬럼 포함)
    
    Returns:
        feature가 추가된 데이터프레임
    """
    questions = questions.copy()
    
    # 텍스트 전처리
    questions["clean_question"] = questions["question"].fillna("").apply(clean_text)
    
    # 문의 유형 분류
    questions["question_type"] = questions["clean_question"].apply(classify_question_type)
    
    # 각 유형별 boolean
    questions["is_delivery"] = (questions["question_type"] == "배송")
    questions["is_refund"] = (questions["question_type"] == "환불")
    questions["is_defect"] = (questions["question_type"] == "불량")
    questions["is_authenticity"] = (questions["question_type"] == "진위")
    questions["is_option"] = (questions["question_type"] == "옵션")
    questions["is_usage"] = (questions["question_type"] == "사용법")
    
    return questions


def aggregate_question_by_seller(questions: pd.DataFrame, products: pd.DataFrame):
    """
    판매자 단위로 문의 feature 집계
    
    문서의 7.2 상품 문의 Feature 전체 구현:
    - 빈도: 상품당 문의 수, 문의/리뷰 비율
    - 유형: 환불/정품/불량 문의 비율
    
    Args:
        questions: add_question_features가 적용된 문의 데이터프레임
        products: 상품 데이터프레임 (product_id, vendor_name 포함)
    
    Returns:
        판매자 단위 집계 데이터프레임
    """
    # product_id → vendor_name 매핑
    prod_vendor = products[["product_id", "vendor_name"]].drop_duplicates()
    merged = questions.merge(prod_vendor, on="product_id", how="left")
    
    # 판매자별 집계
    result = []
    
    for vendor, g in merged.groupby("vendor_name"):
        total_questions = len(g)
        
        # 유형별 비율 계산
        refund_ratio = g["is_refund"].sum() / total_questions if total_questions > 0 else 0
        defect_ratio = g["is_defect"].sum() / total_questions if total_questions > 0 else 0
        authenticity_ratio = g["is_authenticity"].sum() / total_questions if total_questions > 0 else 0
        
        feature = {
            "vendor_name": vendor,
            "question_count": total_questions,
            "refund_question_ratio": refund_ratio,
            "defect_question_ratio": defect_ratio,
            "authenticity_question_ratio": authenticity_ratio,
        }
        
        result.append(feature)
    
    return pd.DataFrame(result)


def calculate_question_density(seller_df: pd.DataFrame, products: pd.DataFrame):
    """
    핵심 가설 검증 변수: 문의_밀도 = 총_문의_수 / 등록_상품_수
    
    Args:
        seller_df: aggregate_question_by_seller 결과 (question_count 포함)
        products: 상품 데이터프레임 (vendor_name 포함)
    
    Returns:
        question_density 컬럼이 추가된 데이터프레임
    """
    seller_df = seller_df.copy()
    
    # 판매자별 상품 수 계산
    product_counts = products.groupby("vendor_name").size().reset_index(name="product_count")
    
    # 병합
    seller_df = seller_df.merge(product_counts, on="vendor_name", how="left")
    
    # 문의 밀도 계산
    seller_df["product_count"] = seller_df["product_count"].fillna(0)
    seller_df["question_density"] = np.where(
        seller_df["product_count"] > 0,
        seller_df["question_count"] / seller_df["product_count"],
        0
    )
    
    return seller_df


def calculate_question_review_ratio(seller_df: pd.DataFrame):
    """
    문의/리뷰 비율 계산: 문의수 / 리뷰수
    
    리뷰 대비 문의가 비정상적으로 많으면 이상 패턴
    
    Args:
        seller_df: question_count, review_count 포함
    
    Returns:
        question_review_ratio 컬럼이 추가된 데이터프레임
    """
    seller_df = seller_df.copy()
    
    # question_count가 없으면 0으로 처리
    if "question_count" not in seller_df.columns:
        seller_df["question_count"] = 0
    
    # review_count가 없으면 0으로 처리
    if "review_count" not in seller_df.columns:
        seller_df["review_count"] = 0
    
    # 문의/리뷰 비율 (리뷰가 0이면 문의만 있는 것으로 간주하여 무한대 방지)
    seller_df["question_review_ratio"] = np.where(
        seller_df["review_count"] > 0,
        seller_df["question_count"] / seller_df["review_count"],
        seller_df["question_count"]  # 리뷰가 없으면 문의 수 자체를 비율로
    )
    
    return seller_df


def merge_question_features_to_seller(seller_df: pd.DataFrame, question_features: pd.DataFrame):
    """
    문의 feature를 판매자 데이터에 병합
    
    Args:
        seller_df: 판매자 기본 데이터 (vendor_name 포함)
        question_features: aggregate_question_by_seller 결과
    
    Returns:
        문의 feature가 병합된 데이터프레임
    """
    seller_df = seller_df.copy()
    
    # 병합
    seller_df = seller_df.merge(question_features, on="vendor_name", how="left")
    
    # 문의가 없는 판매자는 0으로 채움
    question_cols = [
        "question_count", "refund_question_ratio", 
        "defect_question_ratio", "authenticity_question_ratio"
    ]
    
    for col in question_cols:
        if col in seller_df.columns:
            seller_df[col] = seller_df[col].fillna(0)
    
    return seller_df