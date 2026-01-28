"""
모든 인터랙티브 그래프를 하나의 HTML 대시보드로 생성
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

print("=" * 70)
print("인터랙티브 대시보드 생성 중...")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('output/seller_features.csv')
feature_importance = pd.read_csv('output/feature_importance.csv')
predictions = pd.read_csv('output/prediction_results.csv')

df['label_name'] = df['abusive_label'].map({0: '정상', 1: '악성'})

# Feature 이름 매핑 (전체)
feature_names_kr = {
    'refund_question_ratio': '환불 문의 비율',
    'rating_sentiment_gap': '평점-감성 괴리도',
    'question_review_ratio': '문의/리뷰 비율',
    'defect_question_ratio': '불량 문의 비율',
    'negative_keyword_ratio': '부정 키워드 비율',
    'avg_review_length': '평균 리뷰 길이',
    'review_count': '리뷰 개수',
    'negative_sentiment_ratio': '부정 감성 비율',
    'review_density': '리뷰 밀도',
    'textless_5star_ratio': '텍스트 없는 5점 비율',
    'question_density': '문의 밀도',
    'avg_rating': '평균 평점',
    'rating_std': '평점 표준편차',
    'low_rating_ratio': '저평점 비율',
    'duplicate_review_ratio': '중복 리뷰 비율',
    'question_count': '문의 개수',
    'authenticity_question_ratio': '진품 문의 비율',
    'avg_sentiment_score': '평균 감성 점수',
    'rating_normalized': '정규화 평점',
    'product_count': '상품 개수'
}

# =============================================================================
# 메인 대시보드 생성
# =============================================================================

# 8개 subplot 생성 (4x2)
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        '1. 라벨 분포',
        '2. Feature 중요도',
        '3. 부정 감성 vs 평점 괴리',
        '4. ROC 곡선',
        '5. 리뷰 밀도 분포',
        '6. 문의 밀도 분포',
        '7. 혼동 행렬',
        '8. 조건 충족 분포'
    ),
    specs=[
        [{"type": "pie"}, {"type": "bar"}],
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "box"}, {"type": "box"}],
        [{"type": "heatmap"}, {"type": "bar"}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.15
)

# -----------------------------------------------------------------------------
# 1. 라벨 분포 (파이 차트)
# -----------------------------------------------------------------------------
label_counts = df['label_name'].value_counts()
fig.add_trace(
    go.Pie(
        labels=label_counts.index,
        values=label_counts.values,
        marker=dict(colors=['#2ecc71', '#e74c3c']),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>판매자 수: %{value}명<br>비율: %{percent}<extra></extra>'
    ),
    row=1, col=1
)

# -----------------------------------------------------------------------------
# 2. Feature Importance (Bar)
# -----------------------------------------------------------------------------
top10 = feature_importance.head(10).iloc[::-1]
# Feature 이름을 한글로 변환
feature_names_display = [feature_names_kr.get(f, f) for f in top10['feature']]

fig.add_trace(
    go.Bar(
        x=top10['importance'],
        y=feature_names_display,
        orientation='h',
        marker=dict(color=top10['importance'], colorscale='Viridis'),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}<extra></extra>',
        customdata=top10['feature']  # 원래 이름 저장
    ),
    row=1, col=2
)

# -----------------------------------------------------------------------------
# 3. 산점도 (부정 감성 vs 평점 괴리)
# -----------------------------------------------------------------------------
for label_name, color in [('정상', '#2ecc71'), ('악성', '#e74c3c')]:
    data = df[df['label_name'] == label_name]
    fig.add_trace(
        go.Scatter(
            x=data['negative_sentiment_ratio'],
            y=data['rating_sentiment_gap'],
            mode='markers',
            name=label_name,
            marker=dict(size=8, color=color, opacity=0.7),
            hovertemplate='<b>%{text}</b><br>부정 감성: %{x:.3f}<br>평점 괴리: %{y:.3f}<extra></extra>',
            text=data['vendor_name'],
            showlegend=True
        ),
        row=2, col=1
    )

# -----------------------------------------------------------------------------
# 4. ROC Curve
# -----------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(predictions['actual'], predictions['probability'])
auc = roc_auc_score(predictions['actual'], predictions['probability'])

fig.add_trace(
    go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC={auc:.3f})',
        line=dict(color='darkorange', width=2),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
        showlegend=True
    ),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='navy', dash='dash'),
        showlegend=True
    ),
    row=2, col=2
)

# -----------------------------------------------------------------------------
# 5. 리뷰 밀도 Box Plot
# -----------------------------------------------------------------------------
for label_name, color in [('정상', '#2ecc71'), ('악성', '#e74c3c')]:
    data = df[df['label_name'] == label_name]['review_density']
    fig.add_trace(
        go.Box(
            y=data,
            name=label_name,
            marker_color=color,
            hovertemplate='<b>%{fullData.name}</b><br>리뷰 밀도: %{y:.1f}<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )

# -----------------------------------------------------------------------------
# 6. 문의 밀도 Box Plot
# -----------------------------------------------------------------------------
for label_name, color in [('정상', '#2ecc71'), ('악성', '#e74c3c')]:
    data = df[df['label_name'] == label_name]['question_density']
    fig.add_trace(
        go.Box(
            y=data,
            name=label_name,
            marker_color=color,
            hovertemplate='<b>%{fullData.name}</b><br>문의 밀도: %{y:.1f}<extra></extra>',
            showlegend=False
        ),
        row=3, col=2
    )

# -----------------------------------------------------------------------------
# 7. Confusion Matrix
# -----------------------------------------------------------------------------
cm = confusion_matrix(predictions['actual'], predictions['predicted'])
fig.add_trace(
    go.Heatmap(
        z=cm,
        x=['예측: 정상', '예측: 악성'],
        y=['실제: 정상', '실제: 악성'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        hovertemplate='%{y}<br>%{x}<br>개수: %{z}명<extra></extra>',
        showscale=False
    ),
    row=4, col=1
)

# -----------------------------------------------------------------------------
# 8. 조건 충족 개수 분포
# -----------------------------------------------------------------------------
conditions_dist = df['conditions_met_count'].value_counts().sort_index()
colors_map = {0: '#27ae60', 1: '#2ecc71', 2: '#3498db', 
              3: '#f39c12', 4: '#e67e22', 5: '#e74c3c', 6: '#c0392b'}
bar_colors = [colors_map.get(i, '#95a5a6') for i in conditions_dist.index]

fig.add_trace(
    go.Bar(
        x=conditions_dist.index,
        y=conditions_dist.values,
        marker=dict(color=bar_colors),
        hovertemplate='조건 %{x}개 충족<br>판매자 수: %{y}명<extra></extra>',
        showlegend=False
    ),
    row=4, col=2
)

# =============================================================================
# 레이아웃 설정
# =============================================================================
fig.update_layout(
    title_text="악성 판매자 탐지 - 인터랙티브 대시보드<br><sub></sub>",
    title_font_size=20,
    height=1800,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# 축 레이블
fig.update_xaxes(title_text="부정 감성 비율", row=2, col=1)
fig.update_yaxes(title_text="평점-감성 괴리도", row=2, col=1)

fig.update_xaxes(title_text="False Positive Rate", row=2, col=2)
fig.update_yaxes(title_text="True Positive Rate", row=2, col=2)

fig.update_yaxes(title_text="리뷰 밀도", row=3, col=1)
fig.update_yaxes(title_text="문의 밀도", row=3, col=2)

fig.update_xaxes(title_text="조건 충족 개수", row=4, col=2)
fig.update_yaxes(title_text="판매자 수", row=4, col=2)

# =============================================================================
# HTML 저장
# =============================================================================
output_path = 'output/interactive_dashboard.html'
fig.write_html(output_path)

print(f"\n✅ 대시보드 생성 완료!")
print(f"   파일: {output_path}")
print(f"\n📊 포함된 그래프:")
print("   1. 라벨 분포 (파이 차트)")
print("   2. Feature 중요도 (막대 그래프)")
print("   3. 부정 감성 vs 평점 괴리 (산점도)")
print("   4. ROC 곡선")
print("   5. 리뷰 밀도 분포 (Box Plot)")
print("   6. 문의 밀도 분포 (Box Plot)")
print("   7. 혼동 행렬 (히트맵)")
print("   8. 조건 충족 개수 분포 (막대 그래프)")
print(f"\n💡 Tip: open {output_path}")
print("=" * 70)
