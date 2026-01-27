"""
악성 판매자 라벨 생성 모듈
"""

import numpy as np
import pandas as pd


def calculate_percentile_thresholds(df, column, percentile):
    """
    분위수 기준값 계산
    
    Args:
        df: 데이터프레임
        column: 대상 컬럼명
        percentile: 분위수 (0~100)
    
    Returns:
        기준값 (float)
    """
    if column not in df.columns:
        return None
    return df[column].quantile(percentile / 100)


def label_abusive_seller(df):
    """
    Proxy Label 생성: 6개 조건 중 3개 이상 충족 시 악성 판매자
    
    조건:
    1. 리뷰_밀도 하위 20%
    2. 부정_감성_비율 상위 20%
    3. 환불_문의_비율 상위 20%
    4. 불량_문의_비율 상위 20%
    5. 평점_감성_괴리도 상위 20%
    6. 문의_리뷰_비율 상위 20%
    
    Args:
        df: 판매자 집계 데이터프레임
            필수 컬럼: review_density, negative_sentiment_ratio, 
                      refund_question_ratio, defect_question_ratio,
                      rating_sentiment_gap, question_review_ratio
    
    Returns:
        abusive_label 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    
    # 필수 컬럼 정의
    required_columns = {
        'review_density': '리뷰_밀도',
        'negative_sentiment_ratio': '부정_감성_비율',
        'refund_question_ratio': '환불_문의_비율',
        'defect_question_ratio': '불량_문의_비율',
        'rating_sentiment_gap': '평점_감성_괴리도',
        'question_review_ratio': '문의_리뷰_비율'
    }
    
    # 컬럼 존재 여부 확인
    missing_cols = [col for col in required_columns.keys() if col not in df.columns]
    if missing_cols:
        print(f"경고: 누락된 컬럼이 있습니다: {missing_cols}")
        print("누락된 컬럼은 0으로 채워집니다.")
        for col in missing_cols:
            df[col] = 0
    
    # 분위수 기준값 계산
    review_density_low = calculate_percentile_thresholds(df, 'review_density', 20)  # 하위 20%
    negative_ratio_high = calculate_percentile_thresholds(df, 'negative_sentiment_ratio', 80)  # 상위 20%
    refund_ratio_high = calculate_percentile_thresholds(df, 'refund_question_ratio', 80)
    defect_ratio_high = calculate_percentile_thresholds(df, 'defect_question_ratio', 80)
    gap_high = calculate_percentile_thresholds(df, 'rating_sentiment_gap', 80)
    qr_ratio_high = calculate_percentile_thresholds(df, 'question_review_ratio', 80)
    
    # 6개 조건 평가
    condition_1 = df['review_density'] <= review_density_low if review_density_low is not None else False
    condition_2 = df['negative_sentiment_ratio'] >= negative_ratio_high if negative_ratio_high is not None else False
    condition_3 = df['refund_question_ratio'] >= refund_ratio_high if refund_ratio_high is not None else False
    condition_4 = df['defect_question_ratio'] >= defect_ratio_high if defect_ratio_high is not None else False
    condition_5 = df['rating_sentiment_gap'] >= gap_high if gap_high is not None else False
    condition_6 = df['question_review_ratio'] >= qr_ratio_high if qr_ratio_high is not None else False
    
    # 조건 충족 개수 계산
    conditions_met = (
        condition_1.astype(int) +
        condition_2.astype(int) +
        condition_3.astype(int) +
        condition_4.astype(int) +
        condition_5.astype(int) +
        condition_6.astype(int)
    )
    
    # 라벨 생성: 3개 이상 충족 시 악성(1), 아니면 정상(0)
    df['abusive_label'] = np.where(conditions_met >= 3, 1, 0)
    
    # 디버깅용: 각 판매자가 충족한 조건 개수 저장
    df['conditions_met_count'] = conditions_met
    
    # 통계 출력
    abusive_count = df['abusive_label'].sum()
    total_count = len(df)
    abusive_ratio = abusive_count / total_count * 100 if total_count > 0 else 0
    
    print(f"\n=== 라벨 생성 결과 ===")
    print(f"전체 판매자: {total_count}명")
    print(f"악성 판매자: {abusive_count}명 ({abusive_ratio:.1f}%)")
    print(f"정상 판매자: {total_count - abusive_count}명 ({100 - abusive_ratio:.1f}%)")
    print(f"\n분위수 기준값:")
    print(f"  - 리뷰_밀도 하위 20%: {review_density_low:.3f}" if review_density_low else "  - 리뷰_밀도: 계산 불가")
    print(f"  - 부정_감성_비율 상위 20%: {negative_ratio_high:.3f}" if negative_ratio_high else "  - 부정_감성_비율: 계산 불가")
    print(f"  - 환불_문의_비율 상위 20%: {refund_ratio_high:.3f}" if refund_ratio_high else "  - 환불_문의_비율: 계산 불가")
    print(f"  - 불량_문의_비율 상위 20%: {defect_ratio_high:.3f}" if defect_ratio_high else "  - 불량_문의_비율: 계산 불가")
    print(f"  - 평점_감성_괴리도 상위 20%: {gap_high:.3f}" if gap_high else "  - 평점_감성_괴리도: 계산 불가")
    print(f"  - 문의_리뷰_비율 상위 20%: {qr_ratio_high:.3f}" if qr_ratio_high else "  - 문의_리뷰_비율: 계산 불가")
    
    return df


def analyze_label_distribution(df):
    """
    라벨 분포 상세 분석
    
    Args:
        df: abusive_label과 conditions_met_count가 포함된 데이터프레임
    """
    if 'abusive_label' not in df.columns:
        print("abusive_label 컬럼이 없습니다.")
        return
    
    print("\n=== 조건 충족 개수별 분포 ===")
    if 'conditions_met_count' in df.columns:
        condition_dist = df['conditions_met_count'].value_counts().sort_index()
        for count, freq in condition_dist.items():
            print(f"{count}개 조건 충족: {freq}명")
    
    print("\n=== 클래스 불균형 분석 ===")
    label_counts = df['abusive_label'].value_counts()
    for label, count in label_counts.items():
        label_name = "악성" if label == 1 else "정상"
        ratio = count / len(df) * 100
        print(f"{label_name} 판매자: {count}명 ({ratio:.1f}%)")
    
    # 불균형 정도 계산
    if len(label_counts) == 2:
        imbalance_ratio = max(label_counts) / min(label_counts)
        print(f"\n불균형 비율: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 3:
            print("⚠️ 클래스 불균형이 심합니다. SMOTE 또는 Class Weight 적용을 권장합니다.")