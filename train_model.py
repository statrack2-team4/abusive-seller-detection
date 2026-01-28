import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import shap

# =========================
# ✅ 한글 폰트 설정 (Mac)
# =========================
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font_prop = font_manager.FontProperties(fname=FONT_PATH)
rc("font", family=font_prop.get_name())
plt.rcParams["axes.unicode_minus"] = False
print(f"✅ 사용 폰트: {FONT_PATH}")

# =========================
# ✅ 데이터 불러오기
# =========================
DATA_PATH = "data/train_data.csv"   # 네 실제 경로에 맞게 유지
df = pd.read_csv(DATA_PATH)

# =========================
# ✅ 대시보드 컬럼명 통일
# =========================
TARGET_COL = "악성여부"
DROP_COLS = ["vendor_name"]  # 문자열 컬럼 제거

X = df.drop(columns=[TARGET_COL] + DROP_COLS)
y = df[TARGET_COL]

# =========================
# ✅ train / test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# ✅ LightGBM 모델
# =========================
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# ✅ 예측 & 리포트
# =========================
y_pred = model.predict(X_test)

print("\n📊 분류 리포트")
print(classification_report(y_test, y_pred, target_names=["정상", "악성"]))

# =========================
# ✅ 혼동행렬 시각화
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("혼동행렬")
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.xticks([0, 1], ["정상", "악성"])
plt.yticks([0, 1], ["정상", "악성"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("05_혼동행렬.png", dpi=150)
plt.close()

# =========================
# ✅ Feature Importance
# =========================
importances = model.feature_importances_
features = X.columns

fi_df = pd.DataFrame({
    "특성": features,
    "중요도": importances
}).sort_values(by="중요도", ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(fi_df["특성"], fi_df["중요도"])
plt.gca().invert_yaxis()
plt.title("특성 중요도")
plt.xlabel("중요도")

plt.tight_layout()
plt.savefig("06_특성중요도.png", dpi=150)
plt.close()

# =========================
# ✅ SHAP 값
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

plt.figure()
shap.summary_plot(shap_values[1], X_train, show=False)
plt.tight_layout()
plt.savefig("07_SHAP_요약.png", dpi=150)
plt.close()

print("\n✅ 이미지 저장 완료:")
print("05_혼동행렬.png")
print("06_특성중요도.png")
print("07_SHAP_요약.png")
