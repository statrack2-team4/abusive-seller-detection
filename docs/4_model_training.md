# 4. 모델 학습 (Model Training)

## 개요

이 문서는 어뷰징 판매자 탐지를 위한 머신러닝 모델 학습 과정을 설명합니다. `03_feature_engineering.ipynb`에서 생성된 피처를 기반으로 여러 분류 모델을 학습하고 성능을 비교합니다.

## 1. 데이터 로드

### 1.1 피처 데이터 로드

피처 엔지니어링 단계에서 생성된 최종 피처 데이터를 로드합니다.

```python
FEATURE_PATH = '../data/processed/features.csv'
features_df = pd.read_csv(FEATURE_PATH)
```

**피처 구성:**
- `company_name`: 판매자명 (모델 학습에서 제외)
- `is_abusing_seller`: 타겟 변수 (0: 정상, 1: 어뷰징)
- 기타 모든 컬럼: 학습에 사용되는 피처

### 1.2 피처 목록 확인

```python
exclude_cols = ['company_name', 'is_abusing_seller']
feature_columns = [col for col in features_df.columns if col not in exclude_cols]
```

학습에 사용되는 피처만 추출하여 리스트로 관리합니다.

## 2. 데이터 분할 및 전처리

### 2.1 피처와 타겟 분리

```python
X = features_df[feature_columns]  # 피처
y = features_df['is_abusing_seller'].astype(int)  # 타겟
```

### 2.2 계층적 데이터 분할 (Stratified Split)

총 401개의 데이터를 다음과 같이 분할합니다:

```
전체 데이터 (401개)
├── 나머지 (90%, ~361개)
│   ├── 훈련 세트 (60%, ~241개)
│   └── 테스트 세트 (30%, ~120개)
└── 최후 검증 세트 (10%, ~40개)
```

**1단계: 최후 검증 세트 분리**
```python
X_remain, X_final, y_remain, y_final = train_test_split(
    X, y, 
    test_size=0.1,      # 10%를 최후 검증용으로 분리
    random_state=42, 
    stratify=y          # 클래스 비율 유지
)
```

**2단계: 훈련/테스트 세트 분리**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_remain, y_remain, 
    test_size=1/3,      # 나머지의 1/3 (전체의 30%)
    random_state=42,
    stratify=y_remain
)
```

**Stratified Split의 중요성:**
- 어뷰징 판매자는 전체의 약 10%로 불균형 데이터
- 각 세트에서 동일한 클래스 비율 유지 필요
- `stratify=y` 옵션으로 클래스 비율 보존

### 2.3 스케일링 (Standardization)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**스케일링의 필요성:**
- Logistic Regression, SVM 등은 피처 스케일에 민감
- 평균 0, 표준편차 1로 정규화
- **주의**: 훈련 세트로만 `fit`하고, 테스트 세트는 `transform`만 수행 (데이터 누수 방지)

## 3. 모델 학습

### 3.1 모델 평가 함수

```python
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """모델 평가 및 결과 반환"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'model': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    return results, y_test_pred, y_test_proba
```

**평가 지표:**
- **Train Accuracy**: 훈련 세트 정확도 (과적합 진단용)
- **Test Accuracy**: 테스트 세트 정확도
- **Precision**: 정밀도 (어뷰징으로 예측한 것 중 실제 어뷰징 비율)
- **Recall**: 재현율 (실제 어뷰징 중 올바르게 탐지한 비율)
- **F1-Score**: 정밀도와 재현율의 조화평균
- **ROC-AUC**: ROC 곡선 아래 면적 (전반적인 분류 성능)

### 3.2 Logistic Regression

```python
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
```

**특징:**
- 선형 분류 모델
- 해석이 용이 (계수로 피처 영향도 파악 가능)
- 스케일링된 데이터 사용 필수
- 빠른 학습 속도

**하이퍼파라미터:**
- `max_iter=1000`: 최대 반복 횟수
- `random_state=42`: 재현성을 위한 시드

### 3.3 Random Forest

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
```

**특징:**
- 앙상블 모델 (여러 결정 트리의 투표)
- 스케일링 불필요
- 피처 중요도 제공
- 과적합에 강건

**하이퍼파라미터:**
- `n_estimators=100`: 트리 개수
- `n_jobs=-1`: 모든 CPU 코어 사용
- `random_state=42`: 재현성

### 3.4 Gradient Boosting

```python
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
```

**특징:**
- 순차적 앙상블 (이전 모델의 오류를 다음 모델이 보완)
- 높은 예측 성능
- 스케일링 불필요
- 학습 시간이 긴 편

**하이퍼파라미터:**
- `n_estimators=100`: 부스팅 단계 수
- `random_state=42`: 재현성

### 3.5 성능 비교

세 모델의 성능을 DataFrame으로 정리하여 비교합니다:

```python
results_df = pd.DataFrame(results_list)
```

**비교 기준:**
1. **F1-Score**: 불균형 데이터에서 가장 중요한 지표
2. **ROC-AUC**: 전반적인 분류 능력
3. **Train-Test 정확도 차이**: 과적합 여부 확인

## 주요 학습 포인트

### 데이터 분할 전략
- **3-way split**: Train/Test/Final로 분할하여 최종 검증 세트 확보
- **Stratification**: 불균형 데이터에서 필수
- **일관된 random_state**: 재현 가능한 실험

### 스케일링 주의사항
- 트리 기반 모델(RF, GB)은 스케일링 불필요
- 선형/거리 기반 모델(LR, SVM, KNN)은 스케일링 필수
- **데이터 누수 방지**: 테스트 세트는 transform만 수행

### 평가 지표 선택
- 불균형 데이터에서 Accuracy는 오해의 소지
- **F1-Score**를 주요 지표로 사용
- Precision과 Recall의 trade-off 고려

## 다음 단계

모델 평가 시각화 및 피처 중요도 분석은 `5_model_evaluation.md`에서 다룹니다.
