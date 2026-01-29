"""
analyze_results.py의 모든 그래프를 인터랙티브 대시보드로 생성
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

print("=" * 70)
print("인터랙티브 분석 대시보드 생성 중...")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('output/seller_features.csv')
df['label_name'] = df['abusive_label'].map({0: '정상', 1: '악성'})

# Feature 이름
FEATURE_NAMES = {
    'review_density': '리뷰 밀도',
    'question_density': '문의 밀도',
    'avg_rating': '평균 평점',
    'negative_sentiment_ratio': '부정 감성 비율',
    'rating_sentiment_gap': '평점-감성 괴리도',
    'question_review_ratio': '문의/리뷰 비율'
}

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

print(f"✅ 데이터 로드: {df.shape}")

# =============================================================================
# 메인 대시보드 생성 (4x2 레이아웃)
# =============================================================================

fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        '1. 라벨 분포',
        '2. 조건 충족 개수',
        '3. 리뷰 밀도 비교',
        '4. 문의 밀도 비교',
        '5. 부정 감성 비율 비교',
        '6. 평점-감성 괴리도 비교',
        '7. 부정 감성 vs 평점 괴리 (산점도)',
        '8. 상관관계 히트맵 (주요 Feature)'
    ),
    specs=[
        [{"type": "pie"}, {"type": "bar"}],
        [{"type": "box"}, {"type": "box"}],
        [{"type": "box"}, {"type": "box"}],
        [{"type": "scatter"}, {"type": "heatmap"}]
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.12
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
# 2. 조건 충족 개수 분포
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
    row=1, col=2
)

# 악성 기준선
fig.add_vline(x=2.5, line_dash="dash", line_color="red", row=1, col=2)

# -----------------------------------------------------------------------------
# 3-6. Feature Box Plot (4개)
# -----------------------------------------------------------------------------
box_features = ['review_density', 'question_density', 
                'negative_sentiment_ratio', 'rating_sentiment_gap']
box_positions = [(2, 1), (2, 2), (3, 1), (3, 2)]

normal = df[df['label_name'] == '정상']
abusive = df[df['label_name'] == '악성']

for (feature, (row, col)) in zip(box_features, box_positions):
    # 정상
    fig.add_trace(
        go.Box(
            y=normal[feature],
            name='정상',
            marker_color='#2ecc71',
            boxmean='sd',
            hovertemplate='<b>정상</b><br>' + FEATURE_NAMES[feature] + ': %{y:.3f}<extra></extra>',
            showlegend=(feature == 'review_density'),
            legendgroup='정상'
        ),
        row=row, col=col
    )
    
    # 악성
    fig.add_trace(
        go.Box(
            y=abusive[feature],
            name='악성',
            marker_color='#e74c3c',
            boxmean='sd',
            hovertemplate='<b>악성</b><br>' + FEATURE_NAMES[feature] + ': %{y:.3f}<extra></extra>',
            showlegend=(feature == 'review_density'),
            legendgroup='악성'
        ),
        row=row, col=col
    )

# -----------------------------------------------------------------------------
# 7. 산점도 (부정 감성 vs 평점 괴리)
# -----------------------------------------------------------------------------
for label_name, color in [('정상', '#2ecc71'), ('악성', '#e74c3c')]:
    data = df[df['label_name'] == label_name]
    fig.add_trace(
        go.Scatter(
            x=data['negative_sentiment_ratio'],
            y=data['rating_sentiment_gap'],
            mode='markers',
            name=label_name,
            marker=dict(size=6, color=color, opacity=0.7),
            hovertemplate='<b>%{text}</b><br>' +
                          '부정 감성: %{x:.3f}<br>' +
                          '평점 괴리: %{y:.3f}<br>' +
                          '조건 충족: %{customdata}개<extra></extra>',
            text=data['vendor_name'],
            customdata=data['conditions_met_count'],
            showlegend=False
        ),
        row=4, col=1
    )

# -----------------------------------------------------------------------------
# 8. 상관관계 히트맵 (간소화)
# -----------------------------------------------------------------------------
correlation_features = [
    'review_density', 'question_density', 'negative_sentiment_ratio',
    'rating_sentiment_gap', 'question_review_ratio', 'abusive_label'
]

corr_matrix = df[correlation_features].corr()
corr_labels = [HEATMAP_NAMES.get(f, f) for f in correlation_features]

fig.add_trace(
    go.Heatmap(
        z=corr_matrix.values,
        x=corr_labels,
        y=corr_labels,
        colorscale='RdYlGn_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 9},
        hovertemplate='%{y}<br>%{x}<br>상관계수: %{z:.3f}<extra></extra>',
        showscale=True,
        colorbar=dict(len=0.3, y=0.15)
    ),
    row=4, col=2
)

# =============================================================================
# 레이아웃 설정
# =============================================================================
fig.update_layout(
    title_text="악성 판매자 탐지 분석 대시보드<br><sub>마우스를 올려 상세 정보 확인 | 클릭 드래그로 확대</sub>",
    title_font_size=18,
    height=2000,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="right",
        x=1
    )
)

# Y축 레이블
fig.update_yaxes(title_text="판매자 수", row=1, col=2)
fig.update_yaxes(title_text=FEATURE_NAMES['review_density'], row=2, col=1)
fig.update_yaxes(title_text=FEATURE_NAMES['question_density'], row=2, col=2)
fig.update_yaxes(title_text=FEATURE_NAMES['negative_sentiment_ratio'], row=3, col=1)
fig.update_yaxes(title_text=FEATURE_NAMES['rating_sentiment_gap'], row=3, col=2)
fig.update_yaxes(title_text="평점-감성 괴리도", row=4, col=1)

# X축 레이블
fig.update_xaxes(title_text="조건 충족 개수", row=1, col=2)
fig.update_xaxes(title_text="부정 감성 비율", row=4, col=1)

# 히트맵 축 각도
fig.update_xaxes(tickangle=45, row=4, col=2)

# =============================================================================
# HTML 저장
# =============================================================================
output_path = 'output/analyze_results_dashboard.html'
fig.write_html(output_path)

print(f"\n✅ 대시보드 생성 완료!")
print(f"   파일: {output_path}")
print(f"\n📊 포함된 그래프:")
print("   1. 라벨 분포 (파이 차트)")
print("   2. 조건 충족 개수 분포")
print("   3. 리뷰 밀도 비교 (Box Plot)")
print("   4. 문의 밀도 비교 (Box Plot)")
print("   5. 부정 감성 비율 비교 (Box Plot)")
print("   6. 평점-감성 괴리도 비교 (Box Plot)")
print("   7. 산점도 (부정 감성 vs 평점 괴리)")
print("   8. 상관관계 히트맵")
print(f"\n💡 Tip: open {output_path}")
print("=" * 70)

# =============================================================================
# 추가: 주요 판매자 테이블도 별도 HTML로
# =============================================================================
print("\n추가 분석 생성 중...")

# 의심스러운 판매자
top_suspicious = df.nlargest(10, 'conditions_met_count')[[
    'vendor_name', 'conditions_met_count', 'label_name',
    'negative_sentiment_ratio', 'rating_sentiment_gap', 'avg_rating'
]].round(3)

fig_table = go.Figure(data=[go.Table(
    header=dict(
        values=['판매자명', '조건 충족', '라벨', '부정 감성', '평점 괴리', '평균 평점'],
        fill_color='#e74c3c',
        font=dict(color='white', size=14),
        align='center',
        height=40
    ),
    cells=dict(
        values=[top_suspicious[col] for col in top_suspicious.columns],
        fill_color=[['#ffe6e6' if i % 2 == 0 else 'white' for i in range(len(top_suspicious))]],
        align='left',
        font=dict(size=12),
        height=35
    )
)])

fig_table.update_layout(
    title='가장 의심스러운 판매자 Top 10',
    height=500
)

table_path = 'output/suspicious_sellers_table.html'
fig_table.write_html(table_path)

print(f"✅ 추가 테이블 저장: {table_path}")
print("\n모든 파일 생성 완료! 🎉")
