"""
어뷰징 판매자 탐지 모델 시연 웹사이트
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# 페이지 설정
st.set_page_config(
    page_title="어뷰징 판매자 탐지 시스템", page_icon="🔍", layout="wide"
)

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Hugging Face 설정 (모델 업로드 후 여기에 repo_id 입력)
HF_REPO_ID = "alrq/abusive-seller-detection"
HF_MODEL_FILENAME = "abusing_detector_tuned_tuned_rf.pkl"

# 피처 컬럼 정의 (모델 학습에 사용된 피처)
FEATURE_COLUMNS = [
    "satisfaction_score", "review_count", "total_product_count",
    "product_count_actual", "price_mean", "price_std", "price_min", "price_max",
    "rating_mean", "rating_std", "review_sum", "review_mean",
    "discount_mean", "discount_max", "shipping_fee_mean", "shipping_days_mean",
    "review_count_actual", "review_rating_mean", "review_rating_std",
    "review_length_mean", "review_length_std", "review_length_max",
    "question_count", "answer_rate"
]

# 피처 한글 이름 매핑
FEATURE_NAMES_KR = {
    "company_name": "회사명",
    "is_abusing_seller": "어뷰징 판매자 여부",
    "satisfaction_score": "고객 만족도",
    "review_count": "리뷰 수",
    "total_product_count": "총 상품 수",
    "product_count_actual": "실제 상품 수",
    "price_mean": "평균 가격",
    "price_std": "가격 표준편차",
    "price_min": "최소 가격",
    "price_max": "최대 가격",
    "rating_mean": "평균 평점",
    "rating_std": "평점 표준편차",
    "review_sum": "리뷰 합계",
    "review_mean": "평균 리뷰",
    "discount_mean": "평균 할인율",
    "discount_max": "최대 할인율",
    "shipping_fee_mean": "평균 배송비",
    "shipping_days_mean": "평균 배송일",
    "review_count_actual": "실제 리뷰 수",
    "review_rating_mean": "리뷰 평균 평점",
    "review_rating_std": "리뷰 평점 표준편차",
    "review_length_mean": "리뷰 평균 길이",
    "review_length_std": "리뷰 길이 표준편차",
    "review_length_max": "리뷰 최대 길이",
    "question_count": "질문 수",
    "answer_rate": "답변율",
}


@st.cache_data(ttl=3600)
def load_features():
    """피처 데이터 로드 (CSV 우선, 없으면 DB에서 생성)"""
    features_path = DATA_DIR / "ml_features.csv"

    if features_path.exists():
        return pd.read_csv(features_path)

    # CSV가 없으면 DB에서 생성 (최초 1회)
    try:
        from src.features.feature_generation import FeatureGenerator
        generator = FeatureGenerator().load_data(from_db=True)
        features_df = generator.generate_legacy_features()

        # CSV로 저장 (다음 로드 시 빠르게)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_path, index=False)
        return features_df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None


@st.cache_resource
def load_model():
    """모델 로드 (로컬 우선, 없으면 Hugging Face에서 다운로드)"""
    model_path = MODELS_DIR / "abusing_detector_tuned_tuned_rf.pkl"

    # 1. 로컬 파일 확인
    if model_path.exists():
        return joblib.load(model_path)

    # 2. Hugging Face Hub에서 다운로드
    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            cache_dir=str(MODELS_DIR / "hf_cache"),
        )
        return joblib.load(downloaded_path)
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}")
        return None


@st.cache_data
def prepare_validation_data(_df):
    """검증 데이터 준비"""
    # 필요한 컬럼만 선택
    available_features = [c for c in FEATURE_COLUMNS if c in _df.columns]

    X = _df[available_features]
    y = _df["is_abusing_seller"].astype(int)

    # Train/Test 분할 (노트북과 동일한 분할)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 인덱스를 사용해 원본 데이터에서 정보 가져오기
    test_df = _df.loc[X_test.index].copy()

    return X_test, y_test, test_df


@st.cache_data
def get_predictions(_model, _X):
    """예측 수행 (캐싱)"""
    y_pred = _model.predict(_X)
    y_proba = _model.predict_proba(_X)[:, 1]
    return y_pred, y_proba


def main():
    st.title("🔍 어뷰징 판매자 탐지 시스템")
    st.markdown("---")

    # 데이터 및 모델 로드
    with st.spinner("데이터 로딩 중..."):
        df = load_features()
        model = load_model()

    if df is None:
        st.error("피처 데이터를 로드할 수 없습니다.")
        st.info("1. 노트북을 실행하여 피처를 생성하거나\n2. `data/processed/ml_features.csv` 파일을 확인해주세요.")
        return

    if model is None:
        st.error("모델을 로드할 수 없습니다.")
        st.info(f"Hugging Face repo 확인: https://huggingface.co/{HF_REPO_ID}")
        return

    # 검증 데이터 준비
    X_test, y_test, test_df = prepare_validation_data(df)

    # 사이드바 - 페이지 선택
    page = st.sidebar.radio("페이지 선택", ["📊 전체 검증 결과", "🔎 개별 판매자 조회"])

    if page == "📊 전체 검증 결과":
        show_dashboard(model, X_test, y_test, test_df)
    else:
        show_individual_search(model, X_test, y_test, test_df, df)


def show_dashboard(model, X_test, y_test, test_df):
    """전체 검증 결과 대시보드"""
    st.header("📊 검증 데이터 전체 결과")

    # 예측 수행 (캐싱됨)
    y_pred, y_proba = get_predictions(model, X_test.values)

    # 메트릭 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # 메트릭 카드 표시
    st.subheader("성능 지표")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("정확도", f"{accuracy:.1%}")
    with col2:
        st.metric("정밀도", f"{precision:.1%}")
    with col3:
        st.metric("재현율", f"{recall:.1%}")
    with col4:
        st.metric("F1-Score", f"{f1:.1%}")
    with col5:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.markdown("---")

    # 두 개의 컬럼으로 차트 배치
    col1, col2 = st.columns(2)

    with col1:
        # 혼동 행렬
        st.subheader("혼동 행렬")
        cm = confusion_matrix(y_test, y_pred)

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["정상 예측", "어뷰징 예측"],
                y=["정상 실제", "어뷰징 실제"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

        # 혼동 행렬 해석
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        - **True Negative (정상→정상)**: {tn}건
        - **False Positive (정상→어뷰징)**: {fp}건
        - **False Negative (어뷰징→정상)**: {fn}건
        - **True Positive (어뷰징→어뷰징)**: {tp}건
        """)

    with col2:
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=f"모델 (AUC={roc_auc:.3f})",
                mode="lines",
                line=dict(color="#636EFA", width=2),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="Random",
                mode="lines",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # 예측 결과 테이블
    st.subheader("검증 데이터 예측 결과")

    results_df = test_df[["company_name"]].copy()
    results_df["실제"] = y_test.map({0: "정상", 1: "어뷰징"}).values
    results_df["예측"] = pd.Series(y_pred, index=y_test.index).map({0: "정상", 1: "어뷰징"}).values
    results_df["어뷰징 확률"] = y_proba
    results_df["정답 여부"] = y_test.values == y_pred
    results_df["정답 여부"] = results_df["정답 여부"].map({True: "✅", False: "❌"})
    results_df.columns = ["판매자명", "실제", "예측", "어뷰징 확률", "정답 여부"]

    # 필터 옵션
    filter_option = st.radio("필터", ["전체", "정답만", "오답만"], horizontal=True)

    if filter_option == "정답만":
        results_df = results_df[results_df["정답 여부"] == "✅"]
    elif filter_option == "오답만":
        results_df = results_df[results_df["정답 여부"] == "❌"]

    st.dataframe(
        results_df.style.format({"어뷰징 확률": "{:.2%}"}),
        use_container_width=True,
        height=400,
    )

    # 요약 통계
    total = len(y_test)
    correct = (y_test.values == y_pred).sum()
    st.info(f"검증 데이터 총 {total}건 중 {correct}건 정답 ({correct / total:.1%})")


def show_individual_search(model, X_test, y_test, test_df, full_df):
    """개별 판매자 조회"""
    st.header("🔎 개별 판매자 조회")

    # 판매자 선택
    seller_names = test_df["company_name"].tolist()
    selected_seller = st.selectbox("판매자 선택", seller_names)

    if selected_seller:
        # 선택된 판매자 데이터
        seller_idx = test_df[test_df["company_name"] == selected_seller].index[0]
        seller_features = X_test.loc[seller_idx]
        actual_label = y_test.loc[seller_idx]

        # 예측
        pred = model.predict(seller_features.values.reshape(1, -1))[0]
        proba = model.predict_proba(seller_features.values.reshape(1, -1))[0]

        st.markdown("---")

        # 예측 결과 표시
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("실제 레이블")
            if actual_label == 1:
                st.error("🚨 어뷰징 판매자")
            else:
                st.success("✅ 정상 판매자")

        with col2:
            st.subheader("모델 예측")
            if pred == 1:
                st.error("🚨 어뷰징 판매자")
            else:
                st.success("✅ 정상 판매자")

        with col3:
            st.subheader("정답 여부")
            if actual_label == pred:
                st.success("✅ 정답!")
            else:
                st.error("❌ 오답")

        st.markdown("---")

        # 어뷰징 확률 게이지
        st.subheader("어뷰징 확률")

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                title={"text": "어뷰징 확률 (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if proba[1] > 0.5 else "darkgreen"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 70], "color": "lightyellow"},
                        {"range": [70, 100], "color": "lightcoral"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # 피처 상세 정보
        st.subheader("판매자 피처 정보")

        # 피처를 카테고리별로 그룹화
        feature_groups = {
            "기본 정보": [
                "satisfaction_score",
                "review_count",
                "total_product_count",
                "product_count_actual",
            ],
            "가격 정보": ["price_mean", "price_std", "price_min", "price_max"],
            "평점 정보": ["rating_mean", "rating_std"],
            "리뷰 정보": [
                "review_sum",
                "review_mean",
                "review_count_actual",
                "review_rating_mean",
                "review_rating_std",
                "review_length_mean",
                "review_length_std",
                "review_length_max",
            ],
            "할인/배송 정보": [
                "discount_mean",
                "discount_max",
                "shipping_fee_mean",
                "shipping_days_mean",
            ],
            "고객 문의": ["question_count", "answer_rate"],
        }

        for group_name, features in feature_groups.items():
            with st.expander(group_name, expanded=True):
                group_data = []
                for feat in features:
                    if feat not in seller_features.index:
                        continue
                    value = seller_features[feat]
                    # 값 포맷팅
                    if "rate" in feat or "discount" in feat:
                        formatted_value = (
                            f"{value:.1%}" if value <= 1 else f"{value:.1f}%"
                        )
                    elif "price" in feat or "fee" in feat:
                        formatted_value = f"₩{value:,.0f}"
                    elif isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)

                    group_data.append(
                        {
                            "피처": FEATURE_NAMES_KR.get(feat, feat),
                            "값": formatted_value,
                        }
                    )

                if group_data:
                    st.table(pd.DataFrame(group_data))

        # 피처 중요도 (모델이 Random Forest인 경우)
        if hasattr(model, "feature_importances_"):
            st.markdown("---")
            st.subheader("피처 중요도 (모델 기준)")

            available_features = [c for c in FEATURE_COLUMNS if c in X_test.columns]
            importance_df = pd.DataFrame(
                {"feature": available_features, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=True)

            fig_importance = go.Figure(
                data=go.Bar(
                    x=importance_df["importance"],
                    y=[FEATURE_NAMES_KR.get(f, f) for f in importance_df["feature"]],
                    orientation="h",
                    marker_color="#636EFA",
                )
            )
            fig_importance.update_layout(
                height=600, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="중요도"
            )
            st.plotly_chart(fig_importance, use_container_width=True)


if __name__ == "__main__":
    main()
