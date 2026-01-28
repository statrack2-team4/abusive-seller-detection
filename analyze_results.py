"""
악성 판매자 탐지 결과 분석 (한글 완전 해결 버전)
FontProperties로 직접 폰트 지정
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 직접 폰트 속성 설정
font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)

# 기본 설정
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print(f"✅ 한글 폰트 직접 로드: {font_path}")

# Feature 이름 (Box Plot용)
FEATURE_NAMES = {
    'review_density': '리뷰 밀도',
    'question_density': '문의 밀도', 
    'avg_rating': '평균 평점',
    'negative_sentiment_ratio': '부정 감성 비율',
    'rating_sentiment_gap': '평점-감성 괴리도',
    'question_review_ratio': '문의/리뷰 비율'
}

# Feature 이름 (히트맵용 - 전체)
HEATMAP_NAMES = {
    'review_density': '리뷰 밀도',
    'question_density': '문의 밀도',
    'avg_rating': '평균 평점',
    'rating_std': '평점 표준편차',
    'negative_sentiment_ratio': '부정 감성 비율',
    'rating_sentiment_gap': '평점-감성 괴리도',
    'question_review_ratio': '문의/리뷰 비율',
    'abusive_label': '악성 라벨'
}

LABEL_NAMES = {0: '정상', 1: '악성'}


def set_korean_font(ax):
    """축, 제목, 레이블에 한글 폰트 적용"""
    # 제목
    if ax.get_title():
        ax.set_title(ax.get_title(), fontproperties=fontprop)
    # X축 레이블
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontproperties=fontprop)
    # Y축 레이블
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontproperties=fontprop)
    # X축 눈금 레이블
    for label in ax.get_xticklabels():
        label.set_fontproperties(fontprop)
    # Y축 눈금 레이블
    for label in ax.get_yticklabels():
        label.set_fontproperties(fontprop)
    # 범례
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(fontprop)


def load_data():
    """데이터 로드"""
    print("\n" + "=" * 70)
    print("데이터 로드 중...")
    print("=" * 70)
    
    df = pd.read_csv('output/seller_features.csv')
    
    print(f"\n✅ 데이터 크기: {df.shape}")
    print(f"   - 판매자: {len(df)}명")
    print(f"   - Feature: {len(df.columns)}개")
    
    return df


def analyze_labels(df):
    """라벨 분포 분석"""
    print("\n" + "=" * 70)
    print("라벨 분포 분석")
    print("=" * 70)
    
    label_counts = df['abusive_label'].value_counts().sort_index()
    
    for label, count in label_counts.items():
        label_name = LABEL_NAMES[label]
        percentage = count / len(df) * 100
        print(f"{label_name} 판매자: {count}명 ({percentage:.1f}%)")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    labels = [LABEL_NAMES[l] for l in label_counts.index]
    colors = ['#2ecc71', '#e74c3c']
    
    # 파이 차트
    wedges, texts, autotexts = axes[0].pie(label_counts.values, labels=labels, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
    for text in texts:
        text.set_fontproperties(fontprop)
    axes[0].set_title('판매자 라벨 분포', fontproperties=fontprop, fontsize=14, fontweight='bold')
    
    # 막대 차트
    bars = axes[1].bar(labels, label_counts.values, color=colors, alpha=0.7)
    axes[1].set_ylabel('판매자 수', fontproperties=fontprop, fontsize=12)
    axes[1].set_title('판매자 라벨 분포', fontproperties=fontprop, fontsize=14, fontweight='bold')
    set_korean_font(axes[1])
    
    for i, v in enumerate(label_counts.values):
        axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/01_label_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✅ 그래프 저장: output/01_label_distribution.png")
    plt.show()
    plt.close()


def analyze_conditions(df):
    """조건 충족 개수 분석"""
    print("\n" + "=" * 70)
    print("조건 충족 개수 분석")
    print("=" * 70)
    
    conditions_dist = df['conditions_met_count'].value_counts().sort_index()
    
    print("\n조건별 분포:")
    for count, freq in conditions_dist.items():
        status = LABEL_NAMES[1] if count >= 3 else LABEL_NAMES[0]
        print(f"  {count}개 충족: {freq}명 ({status})")
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = {0: '#27ae60', 1: '#2ecc71', 2: '#3498db', 
                  3: '#f39c12', 4: '#e67e22', 5: '#e74c3c', 6: '#c0392b'}
    bar_colors = [colors_map.get(i, '#95a5a6') for i in conditions_dist.index]
    
    bars = ax.bar(conditions_dist.index, conditions_dist.values, color=bar_colors, alpha=0.7)
    ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='악성 기준선 (3개 이상)')
    
    ax.set_xlabel('충족한 조건 개수', fontproperties=fontprop, fontsize=12)
    ax.set_ylabel('판매자 수', fontproperties=fontprop, fontsize=12)
    ax.set_title('악성 판매자 조건 충족 개수 분포', fontproperties=fontprop, fontsize=14, fontweight='bold')
    set_korean_font(ax)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}명', ha='center', va='bottom', fontweight='bold',
                fontproperties=fontprop)
    
    plt.tight_layout()
    plt.savefig('output/02_conditions_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✅ 그래프 저장: output/02_conditions_distribution.png")
    plt.show()
    plt.close()


def compare_features(df):
    """정상 vs 악성 Feature 비교"""
    print("\n" + "=" * 70)
    print("정상 vs 악성 Feature 비교")
    print("=" * 70)
    
    key_features = [
        'review_density', 'question_density', 'avg_rating',
        'negative_sentiment_ratio', 'rating_sentiment_gap', 'question_review_ratio'
    ]
    
    normal = df[df['abusive_label'] == 0]
    abusive = df[df['abusive_label'] == 1]
    
    if len(normal) == 0 or len(abusive) == 0:
        print("⚠️ 비교할 두 그룹이 없습니다.")
        return
    
    # 통계
    print("\n평균값 비교:")
    comparison = pd.DataFrame({
        'Feature': key_features,
        LABEL_NAMES[0]: [normal[f].mean() for f in key_features],
        LABEL_NAMES[1]: [abusive[f].mean() for f in key_features],
        '차이': [abusive[f].mean() - normal[f].mean() for f in key_features]
    })
    print(comparison.round(3).to_string(index=False))
    
    # Box Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        df_plot = df[['abusive_label', feature]].copy()
        df_plot['라벨'] = df_plot['abusive_label'].map(LABEL_NAMES)
        
        sns.boxplot(data=df_plot, x='라벨', y=feature, ax=ax,
                    palette={LABEL_NAMES[0]: '#2ecc71', LABEL_NAMES[1]: '#e74c3c'})
        
        ax.set_xlabel('판매자 유형', fontproperties=fontprop, fontsize=11)
        ax.set_ylabel(FEATURE_NAMES[feature], fontproperties=fontprop, fontsize=11)
        ax.set_title(f'{FEATURE_NAMES[feature]} 비교', fontproperties=fontprop, 
                     fontsize=12, fontweight='bold')
        set_korean_font(ax)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/03_feature_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ 그래프 저장: output/03_feature_comparison.png")
    plt.show()
    plt.close()


def correlation_analysis(df):
    """상관관계 분석"""
    print("\n" + "=" * 70)
    print("Feature 상관관계 분석")
    print("=" * 70)
    
    correlation_features = [
        'review_density', 'question_density', 'avg_rating', 'rating_std',
        'negative_sentiment_ratio', 'rating_sentiment_gap', 
        'question_review_ratio', 'abusive_label'
    ]
    
    corr_matrix = df[correlation_features].corr()
    
    # 컬럼명을 한글로 변경
    corr_matrix_kr = corr_matrix.copy()
    corr_matrix_kr.columns = [HEATMAP_NAMES[col] for col in corr_matrix.columns]
    corr_matrix_kr.index = [HEATMAP_NAMES[col] for col in corr_matrix.index]
    
    # 히트맵
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix_kr, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature 상관관계 히트맵', fontproperties=fontprop, fontsize=14, 
                 fontweight='bold', pad=20)
    
    # 축 레이블에 한글 폰트 적용
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=fontprop, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=fontprop, rotation=0)
    
    plt.tight_layout()
    plt.savefig('output/04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("\n✅ 그래프 저장: output/04_correlation_heatmap.png")
    plt.show()
    plt.close()
    
    # 라벨과의 상관관계
    print("\n악성 라벨과의 상관관계 (절대값 높은 순):")
    label_corr = corr_matrix['abusive_label'].drop('abusive_label').abs().sort_values(ascending=False)
    for feature, corr_val in label_corr.items():
        actual_corr = corr_matrix.loc[feature, 'abusive_label']
        print(f"  {feature:30s}: {actual_corr:+.3f}")


def top_sellers(df):
    """상위/하위 판매자"""
    print("\n" + "=" * 70)
    print("주요 판매자 분석")
    print("=" * 70)
    
    print("\n=== 가장 의심스러운 판매자 (Top 10) ===")
    top_suspicious = df.nlargest(10, 'conditions_met_count')[[
        'vendor_name', 'conditions_met_count', 'abusive_label',
        'review_density', 'negative_sentiment_ratio', 'avg_rating'
    ]]
    print(top_suspicious.to_string(index=False))
    
    print("\n=== 가장 건전한 판매자 (Top 10) ===")
    top_healthy = df.nsmallest(10, 'conditions_met_count')[[
        'vendor_name', 'conditions_met_count', 'abusive_label',
        'review_density', 'negative_sentiment_ratio', 'avg_rating'
    ]]
    print(top_healthy.to_string(index=False))


def save_summary(df):
    """결과 저장"""
    print("\n" + "=" * 70)
    print("분석 결과 저장")
    print("=" * 70)
    
    normal = df[df['abusive_label'] == 0]
    abusive = df[df['abusive_label'] == 1]
    
    if len(normal) > 0 and len(abusive) > 0:
        key_features = [
            'review_density', 'question_density', 'avg_rating',
            'negative_sentiment_ratio', 'rating_sentiment_gap', 'question_review_ratio'
        ]
        
        summary = pd.DataFrame({
            'Feature': key_features,
            '정상_평균': [normal[f].mean() for f in key_features],
            '악성_평균': [abusive[f].mean() for f in key_features],
            '차이': [abusive[f].mean() - normal[f].mean() for f in key_features]
        })
        
        summary.to_csv('output/feature_comparison.csv', index=False, encoding='utf-8-sig')
        print("✅ 비교 결과 저장: output/feature_comparison.csv")
    
    overall_stats = df.describe().T
    overall_stats.to_csv('output/overall_statistics.csv', encoding='utf-8-sig')
    print("✅ 전체 통계 저장: output/overall_statistics.csv")


def main():
    """메인"""
    print("\n" + "=" * 70)
    print("악성 판매자 탐지 결과 분석 시작")
    print("=" * 70)
    
    df = load_data()
    
    analyze_labels(df)
    analyze_conditions(df)
    compare_features(df)
    correlation_analysis(df)
    top_sellers(df)
    save_summary(df)
    
    print("\n" + "=" * 70)
    print("✅ 모든 분석 완료!")
    print("=" * 70)
    print("\n생성된 파일:")
    print("  - output/01_label_distribution.png")
    print("  - output/02_conditions_distribution.png")
    print("  - output/03_feature_comparison.png")
    print("  - output/04_correlation_heatmap.png")
    print("  - output/feature_comparison.csv")
    print("  - output/overall_statistics.csv")
    print("\n그래프를 확인하세요! 🎉")


if __name__ == "__main__":
    main()