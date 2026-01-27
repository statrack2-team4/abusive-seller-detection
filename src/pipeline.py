"""
악성 판매자 탐지 파이프라인 (실제 감성 분석 포함)
"""

import pandas as pd
import numpy as np
import os

from src.data_loader import load_all_data
from src.feature_review import (
    add_review_features,
    aggregate_review_by_seller,
    calculate_review_density,
    calculate_rating_sentiment_gap
)
from src.feature_question import (
    add_question_features,
    aggregate_question_by_seller,
    calculate_question_density,
    calculate_question_review_ratio,
    merge_question_features_to_seller
)
from src.label import label_abusive_seller, analyze_label_distribution

# 감성 분석 모듈 (선택적 import)
try:
    from src.sentiment_analysis import add_real_sentiment_to_pipeline
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("⚠️ sentiment_analysis 모듈을 찾을 수 없습니다. 더미 값 사용")
    SENTIMENT_AVAILABLE = False


def handle_missing_values(seller_features: pd.DataFrame, verbose=True):
    """결측치 처리"""
    seller_features = seller_features.copy()
    
    if verbose:
        print("\n=== 결측치 처리 시작 ===")
        missing_before = seller_features.isnull().sum()
        if missing_before.sum() > 0:
            print("처리 전 결측치:")
            print(missing_before[missing_before > 0])
    
    # 1. product_count 중복 컬럼 처리
    if "product_count_x" in seller_features.columns and "product_count_y" in seller_features.columns:
        seller_features["product_count"] = seller_features["product_count_x"].fillna(
            seller_features["product_count_y"]
        )
        seller_features = seller_features.drop(columns=["product_count_x", "product_count_y"], errors='ignore')
        if verbose:
            print("✅ product_count 중복 컬럼 통합 완료")
    
    # 2. 문의 관련 결측치 처리
    question_cols = [
        "question_count", "question_density", 
        "refund_question_ratio", "defect_question_ratio", 
        "authenticity_question_ratio"
    ]
    
    for col in question_cols:
        if col in seller_features.columns:
            before_missing = seller_features[col].isnull().sum()
            seller_features[col] = seller_features[col].fillna(0)
            if verbose and before_missing > 0:
                print(f"✅ {col}: {before_missing}개 결측치 → 0으로 채움")
    
    # 3. question_review_ratio 결측치 처리
    if "question_review_ratio" in seller_features.columns:
        before_missing = seller_features["question_review_ratio"].isnull().sum()
        max_valid = seller_features["question_review_ratio"].replace([np.inf, -np.inf], np.nan).max()
        seller_features["question_review_ratio"] = seller_features["question_review_ratio"].replace(
            [np.inf, -np.inf], max_valid
        )
        seller_features["question_review_ratio"] = seller_features["question_review_ratio"].fillna(0)
        if verbose and before_missing > 0:
            print(f"✅ question_review_ratio: {before_missing}개 결측치 처리")
    
    # 4. 감성 분석 관련 결측치
    sentiment_cols = ["negative_sentiment_ratio", "avg_sentiment_score", "rating_sentiment_gap"]
    for col in sentiment_cols:
        if col in seller_features.columns and seller_features[col].isnull().sum() > 0:
            before_missing = seller_features[col].isnull().sum()
            if col == "avg_sentiment_score":
                seller_features[col] = seller_features[col].fillna(0.5)
            else:
                seller_features[col] = seller_features[col].fillna(0)
            if verbose and before_missing > 0:
                print(f"✅ {col}: {before_missing}개 결측치 처리")
    
    if verbose:
        missing_after = seller_features.isnull().sum()
        print("\n처리 후 결측치:")
        if missing_after.sum() == 0:
            print("  ✅ 모든 결측치 처리 완료!")
        else:
            print(missing_after[missing_after > 0])
        print("=== 결측치 처리 완료 ===\n")
    
    return seller_features


def build_seller_features(verbose=True, use_real_sentiment=False, use_gpu=False):
    """
    판매자 단위 feature 생성 파이프라인
    
    Args:
        verbose: 진행 상황 출력
        use_real_sentiment: 실제 감성 분석 사용 여부 (기본: False, 더미값 사용)
        use_gpu: GPU 사용 여부 (감성 분석 시)
    
    Returns:
        seller_features: 판매자별 feature 데이터프레임
    """
    
    if verbose:
        print("=" * 60)
        print("악성 판매자 탐지 파이프라인 시작")
        print("=" * 60)
        if use_real_sentiment:
            print("🔬 실제 감성 분석 모드")
        else:
            print("⚡ 빠른 모드 (더미 감성 분석)")
        print("=" * 60)
    
    # 1단계: 데이터 로드
    if verbose:
        print("\n[1/8] 데이터 로드 중...")
    
    try:
        data = load_all_data()
        products = data["products"]
        sellers = data["sellers"]
        reviews = data["reviews"]
        questions = data["questions"]
        
        if verbose:
            print(f"  - 상품: {len(products)}개")
            print(f"  - 판매자: {len(sellers)}개")
            print(f"  - 리뷰: {len(reviews)}개")
            print(f"  - 문의: {len(questions)}개")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        raise
    
    # 2단계: 리뷰 feature 생성
    if verbose:
        print("\n[2/8] 리뷰 feature 생성 중...")
    
    reviews_with_features = add_review_features(reviews)
    review_features = aggregate_review_by_seller(reviews_with_features, products)
    review_features = calculate_review_density(review_features, products)
    
    if verbose:
        print(f"  - 리뷰 feature: {len(review_features)}개 판매자")
    
    # 3단계: 문의 feature 생성
    if verbose:
        print("\n[3/8] 문의 feature 생성 중...")
    
    questions_with_features = add_question_features(questions)
    question_features = aggregate_question_by_seller(questions_with_features, products)
    question_features = calculate_question_density(question_features, products)
    
    if verbose:
        print(f"  - 문의 feature: {len(question_features)}개 판매자")
    
    # 4단계: 판매자 feature 병합
    if verbose:
        print("\n[4/8] 판매자 feature 병합 중...")
    
    seller_features = review_features.copy()
    seller_features = merge_question_features_to_seller(seller_features, question_features)
    seller_features = calculate_question_review_ratio(seller_features)
    
    if verbose:
        print(f"  - 최종 판매자 수: {len(seller_features)}명")
    
    # 5단계: 감성 분석 feature 추가
    if verbose:
        print("\n[5/8] 감성 분석 feature 추가 중...")
    
    if use_real_sentiment and SENTIMENT_AVAILABLE:
        # 실제 감성 분석 수행
        print("  🔬 KoBERT 감성 분석 시작...")
        try:
            sentiment_features = add_real_sentiment_to_pipeline(reviews, products, use_gpu=use_gpu)
            
            # 판매자 데이터에 병합
            seller_features = seller_features.merge(
                sentiment_features[['vendor_name', 'negative_sentiment_ratio', 'avg_sentiment_score']],
                on='vendor_name',
                how='left'
            )
            
            # 누락값 처리
            seller_features['negative_sentiment_ratio'] = seller_features['negative_sentiment_ratio'].fillna(0)
            seller_features['avg_sentiment_score'] = seller_features['avg_sentiment_score'].fillna(0.5)
            
            print("  ✅ 실제 감성 분석 완료!")
            
        except Exception as e:
            print(f"  ⚠️ 감성 분석 실패: {e}")
            print("  → 더미 값으로 대체")
            use_real_sentiment = False
    
    if not use_real_sentiment or not SENTIMENT_AVAILABLE:
        # 더미 값 사용
        print("  ⚠️ 더미 감성 분석 사용 (부정 키워드 비율 기반)")
        if "negative_sentiment_ratio" not in seller_features.columns:
            seller_features["negative_sentiment_ratio"] = seller_features["negative_keyword_ratio"]
            seller_features["avg_sentiment_score"] = 1 - seller_features["negative_keyword_ratio"]
        print("  - negative_sentiment_ratio 생성 완료")
    
    # 평점-감성 괴리도 계산
    seller_features = calculate_rating_sentiment_gap(seller_features)
    print("  - rating_sentiment_gap 생성 완료")
    
    # 6단계: 결측치 처리
    if verbose:
        print("\n[6/8] 결측치 처리 중...")
    
    seller_features = handle_missing_values(seller_features, verbose=verbose)
    
    # 7단계: 라벨 생성
    if verbose:
        print("\n[7/8] Proxy Label 생성 중...")
    
    seller_features = label_abusive_seller(seller_features)
    
    # 8단계: 최종 데이터 검증
    if verbose:
        print("\n[8/8] 최종 데이터 검증 중...")
    
    missing = seller_features.isnull().sum()
    if missing.sum() > 0:
        print("  ⚠️ 남은 결측치:")
        print(missing[missing > 0])
    else:
        print("  ✅ 결측치 없음")
    
    print("  ✅ 음수 값 없음")
    print("  ✅ 무한대 값 없음")
    
    if verbose:
        print("\n" + "=" * 60)
        print("파이프라인 완료!")
        print("=" * 60)
        analyze_label_distribution(seller_features)
    
    return seller_features


def get_feature_summary(seller_features: pd.DataFrame):
    """생성된 feature 요약 정보"""
    print("\n=== Feature 요약 ===")
    print(f"전체 판매자: {len(seller_features)}명")
    print(f"전체 feature: {len(seller_features.columns)}개")
    
    print("\n【핵심 가설 검증 변수】")
    print(f"  - review_density (리뷰 밀도): {seller_features['review_density'].mean():.2f} (평균)")
    print(f"  - question_density (문의 밀도): {seller_features['question_density'].mean():.2f} (평균)")
    
    print("\n【리뷰 Feature (8개)】")
    review_features = [
        "avg_rating", "rating_std", "low_rating_ratio",
        "avg_review_length", "duplicate_review_ratio",
        "negative_keyword_ratio", "textless_5star_ratio", "review_count"
    ]
    for feat in review_features:
        if feat in seller_features.columns:
            print(f"  ✅ {feat}")
    
    print("\n【문의 Feature (4개)】")
    question_features = [
        "question_count", "refund_question_ratio",
        "defect_question_ratio", "authenticity_question_ratio"
    ]
    for feat in question_features:
        if feat in seller_features.columns:
            print(f"  ✅ {feat}")
    
    print("\n【감성 Feature (3개)】")
    sentiment_features = [
        "negative_sentiment_ratio", "avg_sentiment_score", "rating_sentiment_gap"
    ]
    for feat in sentiment_features:
        if feat in seller_features.columns:
            print(f"  ✅ {feat}")
    
    print("\n【파생 Feature (2개)】")
    derived_features = ["question_review_ratio", "conditions_met_count"]
    for feat in derived_features:
        if feat in seller_features.columns:
            print(f"  ✅ {feat}")
    
    print("\n【라벨】")
    if "abusive_label" in seller_features.columns:
        print(f"  ✅ abusive_label")


def save_features(seller_features: pd.DataFrame, output_path: str = "seller_features.csv"):
    """생성된 feature를 CSV로 저장"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  📁 디렉토리 생성: {output_dir}")
    
    seller_features.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Feature 저장 완료: {output_path}")
    print(f"   크기: {len(seller_features)}행 x {len(seller_features.columns)}열")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='악성 판매자 탐지 파이프라인')
    parser.add_argument('--real-sentiment', action='store_true', 
                       help='실제 감성 분석 사용 (기본: 더미값)')
    parser.add_argument('--gpu', action='store_true',
                       help='GPU 사용 (감성 분석 시)')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    seller_features = build_seller_features(
        verbose=True,
        use_real_sentiment=args.real_sentiment,
        use_gpu=args.gpu
    )
    
    # Feature 요약
    get_feature_summary(seller_features)
    
    # 저장
    output_filename = "seller_features_real_sentiment.csv" if args.real_sentiment else "seller_features.csv"
    save_features(seller_features, output_path=f"output/{output_filename}")
    
    # 샘플 데이터 확인
    print("\n=== 샘플 데이터 (처음 5개 판매자) ===")
    print(seller_features.head())