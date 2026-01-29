# 5. 모델 평가 (Model Evaluation)

## 개요

이 문서는 학습된 모델의 성능을 다양한 시각화와 분석 기법으로 평가하는 과정을 설명합니다. 모델 간 성능 비교, 혼동 행렬, ROC 곡선, 피처 중요도 분석을 포함합니다.

## 4. 모델 평가 시각화

### 4.1 성능 비교 차트

```python
metrics = ['test_accuracy', 'precision', 'recall', 'f1']
metric_names = ['정확도', '정밀도', '재현율', 'F1-Score']

fig = go.Figure()

for _, row in results_df.iterrows():
    fig.add_trace(go.Bar(
        name=row['model'],
        x=metric_names,
        y=[row[m] for m in metrics]
    ))

fig.update_layout(
    title='모델별 성능 비교',
    barmode='group',
    yaxis_title='Score',
    template='plotly_white'
)
```

**시각화 목적:**
- 여러 평가 지표를 한눈에 비교
- 각 모델의 강점과 약점 파악
- 모델 선택 근거 제공

**그룹화된 막대 차트의 장점:**
- 모델별 직접 비교 용이
- 다중 지표 동시 표시
- 상호작용 가능 (Plotly)

### 4.2 혼동 행렬 (Confusion Matrix)

```python
# 최고 성능 모델 선택 (F1 기준)
best_model_name = results_df.loc[results_df['f1'].idxmax(), 'model']

# 예측 수행
y_pred = best_model.predict(X_test_final)
cm = confusion_matrix(y_test, y_pred)

# 히트맵 시각화
fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=['정상 예측', '어뷰징 예측'],
    y=['정상 실제', '어뷰징 실제'],
    text=cm,
    texttemplate='%{text}',
    colorscale='Blues'
))
```

**혼동 행렬 구조:**

```
                예측
              정상  어뷰징
실제 정상     TN    FP
     어뷰징    FN    TP
```

- **TN (True Negative)**: 정상을 정상으로 올바르게 예측
- **FP (False Positive)**: 정상을 어뷰징으로 잘못 예측 (1종 오류)
- **FN (False Negative)**: 어뷰징을 정상으로 잘못 예측 (2종 오류) - **가장 위험**
- **TP (True Positive)**: 어뷰징을 어뷰징으로 올바르게 예측

**어뷰징 탐지 맥락에서:**
- **FN을 최소화하는 것이 중요** (실제 어뷰징을 놓치면 안 됨)
- FP는 상대적으로 덜 치명적 (추가 검토로 필터링 가능)
- Recall을 우선시하되, Precision과의 균형 필요 → F1-Score 중요

### 4.3 ROC 곡선 (Receiver Operating Characteristic)

```python
fig = go.Figure()

for name, model in models.items():
    # 확률 예측
    if name == 'Logistic Regression':
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC 곡선 계산
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'{name} (AUC={auc:.3f})',
        mode='lines'
    ))

# 대각선 (랜덤 분류기)
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    name='Random',
    mode='lines',
    line=dict(dash='dash', color='gray')
))
```

**ROC 곡선 해석:**
- **X축 (FPR)**: False Positive Rate = FP / (FP + TN)
  - 정상을 어뷰징으로 잘못 예측하는 비율
- **Y축 (TPR = Recall)**: True Positive Rate = TP / (TP + FN)
  - 실제 어뷰징을 올바르게 탐지하는 비율
- **AUC (Area Under Curve)**:
  - 1.0에 가까울수록 우수
  - 0.5는 랜덤 분류기 수준
  - 일반적 기준: 0.7-0.8 (양호), 0.8-0.9 (우수), 0.9+ (탁월)

**ROC 곡선의 장점:**
- 임계값(threshold) 변화에 따른 성능 변화 시각화
- 모델 간 전반적인 분류 능력 비교
- 클래스 불균형에도 비교적 안정적

## 5. 피처 중요도 분석

### 5.1 Random Forest 피처 중요도

```python
# 피처 중요도 추출
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

# 수평 막대 차트
fig = go.Figure(data=go.Bar(
    x=feature_importance['importance'],
    y=feature_importance['feature'],
    orientation='h',
    marker_color='#636EFA'
))

fig.update_layout(
    title='Random Forest - 피처 중요도',
    xaxis_title='Importance',
    yaxis_title='Feature',
    height=600
)
```

**피처 중요도 계산 방법:**
- Random Forest는 각 트리에서 피처를 사용할 때 불순도 감소량을 측정
- 모든 트리에 걸쳐 평균을 계산
- 높을수록 모델 예측에 중요한 피처

**활용 방법:**
1. **피처 선택**: 중요도가 낮은 피처 제거로 모델 단순화
2. **도메인 검증**: 중요 피처가 비즈니스 로직과 일치하는지 확인
3. **데이터 수집 우선순위**: 중요한 피처의 데이터 품질 개선에 집중

### 5.2 Top 10 중요 피처 분석

```python
print("\n=== Top 10 중요 피처 ===")
print(feature_importance.tail(10).to_string(index=False))
```

**예상되는 중요 피처 (도메인 지식 기반):**
1. **리뷰 관련 피처**
   - `review_length_mean`: 리뷰 길이 평균
   - `review_rating_mean`: 리뷰 평점 평균
   - 어뷰징 판매자는 짧고 극단적인 평점의 리뷰가 많음

2. **가격 관련 피처**
   - `discount_mean`: 평균 할인율
   - `price_std`: 가격 표준편차
   - 비정상적인 가격 정책 탐지

3. **판매자 활동 지표**
   - `answer_rate`: 질문 답변율
   - `satisfaction_score`: 만족도 점수
   - 어뷰징 판매자는 고객 응대가 부실할 가능성

## 6. 모델 저장

### 6.1 최고 성능 모델 저장

```python
os.makedirs('../models', exist_ok=True)

# 모델 파일 저장
joblib.dump(best_model, f'../models/abusing_detector_{best_model_name.lower().replace(" ", "_")}.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

# 피처 목록 저장
with open('../models/feature_columns.txt', 'w') as f:
    f.write('\n'.join(feature_columns))
```

**저장되는 파일:**
1. **모델 파일**: `abusing_detector_<model_name>.pkl`
   - 학습된 모델 객체
   - 예측 시 직접 로드하여 사용

2. **스케일러**: `scaler.pkl`
   - 훈련 데이터로 학습된 StandardScaler
   - 새로운 데이터 전처리 시 필수

3. **피처 목록**: `feature_columns.txt`
   - 모델 학습 시 사용된 피처 순서
   - 예측 시 동일한 순서로 입력해야 함

### 6.2 모델 저장 시 고려사항

**버전 관리:**
- 날짜 또는 버전 번호를 파일명에 포함
- 예: `abusing_detector_v1.0_20240129.pkl`

**메타데이터 저장:**
```python
metadata = {
    'model_type': best_model_name,
    'training_date': datetime.now().isoformat(),
    'feature_count': len(feature_columns),
    'test_f1': results_df.loc[results_df['f1'].idxmax(), 'f1'],
    'test_roc_auc': results_df.loc[results_df['f1'].idxmax(), 'roc_auc']
}
joblib.dump(metadata, '../models/model_metadata.pkl')
```

**재현성 보장:**
- 데이터 버전
- 하이퍼파라미터
- 랜덤 시드
- 라이브러리 버전 (requirements.txt)

## 모델 선택 기준

### 최종 모델 결정 프로세스

1. **F1-Score 우선 고려** (불균형 데이터)
2. **과적합 여부 확인** (Train-Test 정확도 차이)
3. **ROC-AUC로 전반적 성능 검증**
4. **비즈니스 요구사항 반영**
   - 속도 vs 정확도
   - 해석 가능성 필요 여부
   - 운영 환경 제약

### 트레이드오프 고려

**Logistic Regression:**
- ✅ 빠른 예측 속도
- ✅ 해석 용이
- ❌ 복잡한 패턴 학습 어려움

**Random Forest:**
- ✅ 높은 정확도
- ✅ 과적합 방지
- ❌ 모델 크기 큼
- ❌ 예측 속도 느림

**Gradient Boosting:**
- ✅ 최고 성능
- ✅ 피처 중요도 제공
- ❌ 학습 시간 김
- ❌ 하이퍼파라미터 튜닝 복잡

## 다음 단계

하이퍼파라미터 튜닝과 고급 기법은 `6_hyperparameter_tuning.md`에서 다룹니다.
