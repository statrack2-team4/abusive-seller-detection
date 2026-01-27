"""
리뷰 감성 분석 모듈 (개선된 키워드 기반)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


class SentimentAnalyzer:
    """
    개선된 키워드 기반 감성 분석기
    - 이중부정 처리
    - 감성 강도 가중치
    - 빠른 속도
    """
    
    def __init__(self):
        """초기화"""
        print(f"감성 분석 모듈 초기화 중...")
        print("✅ 개선된 키워드 기반 감성 분석 준비 완료!")
    
    
    def analyze_text(self, text: str) -> dict:
        """
        단일 텍스트 감성 분석
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            {
                'label': 'positive' or 'negative',
                'positive_score': 0~1,
                'negative_score': 0~1
            }
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'label': 'neutral',
                'positive_score': 0.5,
                'negative_score': 0.5
            }
        
        return self._analyze_sentiment(text)
    
    
    def _analyze_sentiment(self, text: str) -> dict:
        """
        개선된 키워드 기반 감성 분석
        
        특징:
        - 이중부정 처리 ("나쁘지 않다" → 긍정)
        - 감성 강도 구분 (강/약)
        - 문맥 고려
        """
        # 강한 긍정 키워드 (가중치 2)
        strong_positive = [
            '최고', '완벽', '훌륭', '강추', '추천', 
            '대만족', '감사', '좋아요', '만족', '굿'
        ]
        
        # 약한 긍정 키워드 (가중치 1)
        weak_positive = [
            '좋', '괜찮', '쓸만', '그럭저럭', '나쁘지않', 
            '무난', '적당', '보통이상'
        ]
        
        # 강한 부정 키워드 (가중치 2)
        strong_negative = [
            '최악', '환불', '불량', '고장', '사기', '먹튀', 
            '실망', '짜증', '화남', '별로'
        ]
        
        # 약한 부정 키워드 (가중치 1)
        weak_negative = [
            '아쉽', '그냥', '보통', '별로', '글쎄',
            '애매', '미흡'
        ]
        
        # 부정 표현 (이중부정 탐지용)
        negation = ['안', '않', '못', '없']
        
        text_lower = text.lower()
        
        # 점수 계산
        pos_score = 0
        neg_score = 0
        
        # 1. 이중부정 체크 ("나쁘지 않다" = 긍정)
        double_negation_patterns = [
            ('나쁘', negation),
            ('별로', negation),
            ('안좋', negation),
            ('불편', negation)
        ]
        
        for neg_word, neg_list in double_negation_patterns:
            for negation_word in neg_list:
                if neg_word in text_lower and negation_word in text_lower:
                    # 이중부정 발견 = 약한 긍정
                    pos_score += 1.5
        
        # 2. 강한 긍정 키워드
        for kw in strong_positive:
            if kw in text_lower:
                pos_score += 2
        
        # 3. 약한 긍정 키워드
        for kw in weak_positive:
            if kw in text_lower:
                pos_score += 1
        
        # 4. 강한 부정 키워드
        for kw in strong_negative:
            if kw in text_lower:
                neg_score += 2
        
        # 5. 약한 부정 키워드
        for kw in weak_negative:
            if kw in text_lower:
                neg_score += 1
        
        # 6. 정규화
        total = pos_score + neg_score
        
        if total == 0:
            # 감성 키워드 없음 = 중립
            return {
                'label': 'neutral',
                'positive_score': 0.5,
                'negative_score': 0.5
            }
        
        positive_score = pos_score / total
        negative_score = neg_score / total
        
        # 라벨 결정
        if positive_score > negative_score:
            label = 'positive'
        elif negative_score > positive_score:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'label': label,
            'positive_score': positive_score,
            'negative_score': negative_score
        }
    
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
        """
        데이터프레임의 텍스트 일괄 감성 분석
        
        Args:
            df: 분석할 데이터프레임
            text_column: 텍스트 컬럼명
        
        Returns:
            sentiment_label, positive_score, negative_score 컬럼이 추가된 데이터프레임
        """
        df = df.copy()
        
        print(f"\n감성 분석 시작... (총 {len(df)}개)")
        
        results = []
        texts = df[text_column].fillna("").tolist()
        
        # 진행바 표시
        for text in tqdm(texts, desc="감성 분석"):
            result = self.analyze_text(text)
            results.append(result)
        
        # 결과를 데이터프레임에 추가
        df['sentiment_label'] = [r['label'] for r in results]
        df['positive_score'] = [r['positive_score'] for r in results]
        df['negative_score'] = [r['negative_score'] for r in results]
        
        print("✅ 감성 분석 완료!")
        
        return df


def aggregate_sentiment_by_seller(reviews_with_sentiment: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    판매자 단위로 감성 분석 결과 집계
    
    Args:
        reviews_with_sentiment: 감성 분석 결과가 포함된 리뷰 데이터
        products: 상품 데이터 (product_id, vendor_name 포함)
    
    Returns:
        판매자별 감성 분석 집계 데이터프레임
    """
    # product_id → vendor_name 매핑
    prod_vendor = products[["product_id", "vendor_name"]].drop_duplicates()
    merged = reviews_with_sentiment.merge(prod_vendor, on="product_id", how="left")
    
    # 판매자별 집계
    result = []
    
    for vendor, g in merged.groupby("vendor_name"):
        total_reviews = len(g)
        
        # 부정 리뷰 비율
        negative_count = (g['sentiment_label'] == 'negative').sum()
        negative_ratio = negative_count / total_reviews if total_reviews > 0 else 0
        
        # 평균 감성 점수 (positive_score 평균)
        avg_sentiment_score = g['positive_score'].mean()
        
        result.append({
            'vendor_name': vendor,
            'negative_sentiment_ratio': negative_ratio,
            'avg_sentiment_score': avg_sentiment_score,
            'total_reviews_analyzed': total_reviews
        })
    
    return pd.DataFrame(result)


def add_real_sentiment_to_pipeline(reviews: pd.DataFrame, products: pd.DataFrame, use_gpu: bool = False) -> pd.DataFrame:
    """
    파이프라인에서 사용할 감성 분석 함수
    
    Args:
        reviews: 리뷰 데이터프레임
        products: 상품 데이터프레임
        use_gpu: GPU 사용 여부 (미사용)
    
    Returns:
        판매자별 감성 분석 결과
    """
    # 감성 분석기 초기화
    analyzer = SentimentAnalyzer()
    
    # 리뷰 감성 분석
    reviews_with_sentiment = analyzer.analyze_dataframe(
        reviews,
        text_column='review_text'
    )
    
    # 판매자별 집계
    sentiment_features = aggregate_sentiment_by_seller(reviews_with_sentiment, products)
    
    return sentiment_features


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 70)
    print("감성 분석 모듈 테스트")
    print("=" * 70)
    
    # 테스트 데이터
    test_reviews = pd.DataFrame({
        'review_text': [
            '정말 좋아요! 강력 추천합니다.',
            '불량품이네요. 환불 요청합니다.',
            '그냥 그래요. 보통입니다.',
            '최악입니다. 다시는 안 삽니다.',
            '완벽한 제품! 만족합니다.',
            '나쁘지 않아요. 괜찮습니다.',  # 이중부정
            '좋지 않네요. 별로예요.',      # 부정
            '생각보다 나쁘지 않네요',      # 이중부정
        ]
    })
    
    # 감성 분석 실행
    analyzer = SentimentAnalyzer()
    
    print("\n" + "=" * 70)
    print("개별 텍스트 분석")
    print("=" * 70)
    
    for idx, text in enumerate(test_reviews['review_text'], 1):
        result = analyzer.analyze_text(text)
        emoji = "😊" if result['label'] == 'positive' else "😞" if result['label'] == 'negative' else "😐"
        print(f"\n{idx}. {text}")
        print(f"   {emoji} {result['label']} (긍정: {result['positive_score']:.2f}, 부정: {result['negative_score']:.2f})")
    
    print("\n" + "=" * 70)
    print("데이터프레임 일괄 분석")
    print("=" * 70)
    
    result_df = analyzer.analyze_dataframe(test_reviews)
    print("\n", result_df[['review_text', 'sentiment_label', 'positive_score', 'negative_score']])
    
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)