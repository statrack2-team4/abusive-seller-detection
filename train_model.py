import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from lightgbm import LGBMClassifier

# =============================================================================
# Mac 한글 폰트 (확정 경로)
# =============================================================================
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

print(f"✅ 사용 폰트: {font_path}")

# =============================================================================
# Feature 이름 한글 매핑 (create_dashboard.py와 동일)
# =============================================================================
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
    'product_count': '상품 개수',
    'conditions_met_count': '조건 충족 개수'
}

# =============================================================================
# 출력 폴더
# =============================================================================
os.makedirs("output", exist_ok=True)

# =============================================================================
# 데이터 로드
# =============================================================================
df = pd.read_csv("output/seller_features.csv")
df["label_name"] = df["abusive_label"].map({0: "정상", 1: "악성"})

# 모델 입력 (문자열 컬럼 제거)
drop_cols = ["vendor_name", "label_name"]
X = df.drop(columns=["abusive_label"] + drop_cols)
y = df["abusive_label"]

# =============================================================================
# 학습 / 테스트 분리
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 모델 학습
# =============================================================================
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# =============================================================================
# 예측
# =============================================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 분류 리포트")
print(classification_report(y_test, y_pred, target_names=["정상", "악성"]))

# =============================================================================
# Feature Importance 저장
# =============================================================================
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "feature_kr": [feature_names_kr.get(f, f) for f in X.columns],
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance.to_csv("output/feature_importance.csv", index=False)

# =============================================================================
# 05. 혼동 행렬
# =============================================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["예측: 정상", "예측: 악성"],
    yticklabels=["실제: 정상", "실제: 악성"]
)
plt.title("혼동 행렬")
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.tight_layout()
plt.savefig("output/05_혼동행렬.png", dpi=300)
plt.close()

# =============================================================================
# 06. ROC 곡선
# =============================================================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC 곡선")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("output/06_ROC곡선.png", dpi=300)
plt.close()

# =============================================================================
# 07. 특성 중요도
# =============================================================================
top20 = feature_importance.head(20).iloc[::-1]

plt.figure(figsize=(8, 10))
plt.barh(top20["feature_kr"], top20["importance"])
plt.title("특성 중요도")
plt.xlabel("중요도")
plt.ylabel("특성")
plt.tight_layout()
plt.savefig("output/07_특성중요도.png", dpi=300)
plt.close()

# =============================================================================
# 08. SHAP 요약
# =============================================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

X_train_kr = X_train.copy()
X_train_kr.columns = [feature_names_kr.get(c, c) for c in X_train.columns]

plt.figure()
shap.summary_plot(shap_values_to_plot, X_train_kr, show=False)
plt.title("SHAP 요약 그래프")
plt.tight_layout()
plt.savefig("output/08_SHAP요약.png", dpi=300)
plt.close()

# =============================================================================
# 예측 결과 저장 (대시보드용)
# =============================================================================
predictions = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "probability": y_prob
})
predictions.to_csv("output/prediction_results.csv", index=False)

print("\n✅ 학습 및 그래프 생성 완료")
print("📁 생성 파일:")
print(" - output/05_혼동행렬.png")
print(" - output/06_ROC곡선.png")
print(" - output/07_특성중요도.png")
print(" - output/08_SHAP요약.png")
print(" - output/feature_importance.csv")
print(" - output/prediction_results.csv")
