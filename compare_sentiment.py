"""
더미 감성 분석 vs 실제 감성 분석 결과 비교
"""

import pandas as pd
import numpy as np

def compare_sentiment_results():
    """
    두 CSV 파일을 비교하여 차이점 출력
    """
    print("=" * 70)
    print("감성 분석 결과 비교")
    print("=" * 70)
    
    # 파일 로드
    try:
        df_dummy = pd.read_csv("output/seller_features.csv")
        print("✅ 더미 감성 분석 파일 로드: output/seller_features.csv")
    except FileNotFoundError:
        print("❌ output/seller_features.csv 파일이 없습니다.")
        return
    
    try:
        df_real = pd.read_csv("output/seller_features_real_sentiment.csv")
        print("✅ 실제 감성 분석 파일 로드: output/seller_features_real_sentiment.csv")
    except FileNotFoundError:
        print("❌ output/seller_features_real_sentiment.csv 파일이 없습니다.")
        print("\n💡 실제 감성 분석을 실행하세요:")
        print("   python -m src.pipeline --real-sentiment")
        return
    
    print(f"\n더미 모드: {len(df_dummy)}개 판매자")
    print(f"실제 모드: {len(df_real)}개 판매자")
    
    # 감성 분석 관련 컬럼 비교
    sentiment_cols = ['negative_sentiment_ratio', 'avg_sentiment_score', 'rating_sentiment_gap']
    
    print("\n" + "=" * 70)
    print("감성 분석 Feature 비교")
    print("=" * 70)
    
    for col in sentiment_cols:
        if col in df_dummy.columns and col in df_real.columns:
            print(f"\n【{col}】")
            print(f"  더미 모드 - 평균: {df_dummy[col].mean():.4f}, 표준편차: {df_dummy[col].std():.4f}")
            print(f"  실제 모드 - 평균: {df_real[col].mean():.4f}, 표준편차: {df_real[col].std():.4f}")
            
            # 차이 계산
            diff = np.abs(df_dummy[col].mean() - df_real[col].mean())
            print(f"  → 차이: {diff:.4f}")
            
            if diff > 0.01:
                print(f"  ✅ 실제 감성 분석이 적용되었습니다!")
            else:
                print(f"  ⚠️ 차이가 거의 없습니다. 실제 감성 분석이 실행되지 않았을 수 있습니다.")
    
    # 라벨 비교
    print("\n" + "=" * 70)
    print("라벨 분포 비교")
    print("=" * 70)
    
    dummy_abusive = df_dummy['abusive_label'].sum()
    real_abusive = df_real['abusive_label'].sum()
    
    print(f"\n더미 모드 - 악성: {dummy_abusive}명 ({dummy_abusive/len(df_dummy)*100:.1f}%)")
    print(f"실제 모드 - 악성: {real_abusive}명 ({real_abusive/len(df_real)*100:.1f}%)")
    print(f"→ 차이: {abs(dummy_abusive - real_abusive)}명")
    
    # 샘플 비교 (상위 5개 판매자)
    print("\n" + "=" * 70)
    print("샘플 비교 (처음 5개 판매자)")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'vendor_name': df_dummy['vendor_name'][:5],
        'dummy_neg_ratio': df_dummy['negative_sentiment_ratio'][:5],
        'real_neg_ratio': df_real['negative_sentiment_ratio'][:5],
        'dummy_label': df_dummy['abusive_label'][:5],
        'real_label': df_real['abusive_label'][:5]
    })
    
    print("\n", comparison.to_string(index=False))
    
    # 라벨 변경된 판매자 찾기
    if len(df_dummy) == len(df_real):
        label_changed = df_dummy[df_dummy['abusive_label'] != df_real['abusive_label']]
        
        if len(label_changed) > 0:
            print(f"\n⚠️ 라벨이 변경된 판매자: {len(label_changed)}명")
            print("\n예시:")
            for idx in label_changed.index[:5]:
                vendor = df_dummy.loc[idx, 'vendor_name']
                old_label = df_dummy.loc[idx, 'abusive_label']
                new_label = df_real.loc[idx, 'abusive_label']
                print(f"  - {vendor}: {old_label} → {new_label}")
        else:
            print("\n✅ 모든 판매자의 라벨이 동일합니다.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_sentiment_results()