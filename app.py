"""
어뷰징 판매자 탐지 모델 시연 웹사이트
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# 페이지 설정
st.set_page_config(
    page_title="어뷰징 판매자 탐지 시스템",
    page_icon="🔍",
    layout="wide"
)

# 피처 컬럼 정의
FEATURE_COLUMNS = [
    'satisfaction_score', 'review_count', 'total_product_count',
    'product_count_actual', 'price_mean', 'price_std', 'price_min', 'price_max',
    'rating_mean', 'rating_std', 'review_sum', 'review_mean',
    'discount_mean', 'discount_max', 'shipping_fee_mean', 'shipping_days_mean',
    'review_count_actual', 'review_rating_mean', 'review_rating_std',
    'review_length_mean', 'review_length_std', 'review_length_max',
    'question_count', 'answer_rate'
]

# 피처 한글 이름 매핑
FEATURE_NAMES_KR = {
    'satisfaction_score': '만족도 점수',
    'review_count': '리뷰 수 (프로필)',
    'total_product_count': '총 상품 수 (프로필)',
    'product_count_actual': '실제 상품 수',
    'price_mean': '평균 가격',
    'price_std': '가격 표준편차',
    'price_min': '최소 가격',
    'price_max': '최대 가격',
    'rating_mean': '평균 평점',
    'rating_std': '평점 표준편차',
    'review_sum': '총 리뷰 수',
    'review_mean': '상품당 평균 리뷰',
    'discount_mean': '평균 할인율',
    'discount_max': '최대 할인율',
    'shipping_fee_mean': '평균 배송비',
    'shipping_days_mean': '평균 배송일',
    'review_count_actual': '실제 리뷰 수',
    'review_rating_mean': '리뷰 평균 평점',
    'review_rating_std': '리뷰 평점 표준편차',
    'review_length_mean': '리뷰 평균 길이',
    'review_length_std': '리뷰 길이 표준편차',
    'review_length_max': '리뷰 최대 길이',
    'question_count': '질문 수',
    'answer_rate': '답변율'
}


@st.cache_data
def load_data():
    """피처 데이터 로드"""
    df = pd.read_csv('data/processed/features.csv')
    return df


@st.cache_resource
def load_model():
    """모델 및 스케일러 로드"""
    model = joblib.load('models/abusing_detector_tuned_tuned_rf.pkl')
    scaler = joblib.load('models/scaler_tuned.pkl')
    return model, scaler


def prepare_validation_data(df):
    """검증 데이터 준비 (노트북과 동일한 분할)"""
    X = df[FEATURE_COLUMNS]
    y = df['is_abusing_seller'].astype(int)

    # 1단계: 전체 -> 나머지(90%) / 최후검증(10%)
    X_remain, X_final, y_remain, y_final = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # 인덱스를 사용해 원본 데이터에서 회사명 가져오기
    final_indices = X_final.index
    final_df = df.loc[final_indices].copy()

    return X_final, y_final, final_df


def main():
    st.title("🔍 어뷰징 판매자 탐지 시스템")
    st.markdown("---")

    # 데이터 및 모델 로드
    try:
        df = load_data()
        model, scaler = load_model()
    except FileNotFoundError as e:
        st.error(f"파일을 찾을 수 없습니다: {e}")
        st.info("먼저 노트북을 실행하여 모델과 데이터를 생성해주세요.")
        return

    # 검증 데이터 준비
    X_final, y_final, final_df = prepare_validation_data(df)

    # 사이드바 - 페이지 선택
    page = st.sidebar.radio(
        "페이지 선택",
        ["📊 전체 검증 결과", "🔎 개별 판매자 조회"]
    )

    if page == "📊 전체 검증 결과":
        show_dashboard(model, X_final, y_final, final_df)
    else:
        show_individual_search(model, X_final, y_final, final_df)


def show_dashboard(model, X_final, y_final, final_df):
    """전체 검증 결과 대시보드"""
    st.header("📊 검증 데이터 전체 결과")

    # 예측 수행
    y_pred = model.predict(X_final)
    y_proba = model.predict_proba(X_final)[:, 1]

    # 메트릭 계산
    accuracy = accuracy_score(y_final, y_pred)
    precision = precision_score(y_final, y_pred)
    recall = recall_score(y_final, y_pred)
    f1 = f1_score(y_final, y_pred)
    roc_auc = roc_auc_score(y_final, y_proba)

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
        cm = confusion_matrix(y_final, y_pred)

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['정상 예측', '어뷰징 예측'],
            y=['정상 실제', '어뷰징 실제'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=False
        ))
        fig_cm.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
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
        fpr, tpr, _ = roc_curve(y_final, y_proba)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'모델 (AUC={roc_auc:.3f})',
            mode='lines',
            line=dict(color='#636EFA', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # 예측 결과 테이블
    st.subheader("검증 데이터 예측 결과")

    results_df = final_df[['company_name']].copy()
    results_df['실제'] = y_final.map({0: '정상', 1: '어뷰징'}).values
    results_df['예측'] = pd.Series(y_pred).map({0: '정상', 1: '어뷰징'}).values
    results_df['어뷰징 확률'] = y_proba
    results_df['정답 여부'] = (y_final.values == y_pred)
    results_df['정답 여부'] = results_df['정답 여부'].map({True: '✅', False: '❌'})
    results_df.columns = ['판매자명', '실제', '예측', '어뷰징 확률', '정답 여부']

    # 필터 옵션
    filter_option = st.radio(
        "필터",
        ["전체", "정답만", "오답만"],
        horizontal=True
    )

    if filter_option == "정답만":
        results_df = results_df[results_df['정답 여부'] == '✅']
    elif filter_option == "오답만":
        results_df = results_df[results_df['정답 여부'] == '❌']

    st.dataframe(
        results_df.style.format({'어뷰징 확률': '{:.2%}'}),
        use_container_width=True,
        height=400
    )

    # 요약 통계
    total = len(y_final)
    correct = (y_final.values == y_pred).sum()
    st.info(f"검증 데이터 총 {total}건 중 {correct}건 정답 ({correct/total:.1%})")


def show_individual_search(model, X_final, y_final, final_df):
    """개별 판매자 조회"""
    st.header("🔎 개별 판매자 조회")

    # 판매자 선택
    seller_names = final_df['company_name'].tolist()
    selected_seller = st.selectbox(
        "판매자 선택",
        seller_names,
        format_func=lambda x: f"{x}"
    )

    if selected_seller:
        # 선택된 판매자 데이터
        seller_idx = final_df[final_df['company_name'] == selected_seller].index[0]
        seller_features = X_final.loc[seller_idx]
        actual_label = y_final.loc[seller_idx]

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

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba[1] * 100,
            title={'text': "어뷰징 확률 (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if proba[1] > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # 피처 상세 정보
        st.subheader("판매자 피처 정보")

        # 피처를 카테고리별로 그룹화
        feature_groups = {
            "기본 정보": ['satisfaction_score', 'review_count', 'total_product_count', 'product_count_actual'],
            "가격 정보": ['price_mean', 'price_std', 'price_min', 'price_max'],
            "평점 정보": ['rating_mean', 'rating_std'],
            "리뷰 정보": ['review_sum', 'review_mean', 'review_count_actual', 'review_rating_mean',
                       'review_rating_std', 'review_length_mean', 'review_length_std', 'review_length_max'],
            "할인/배송 정보": ['discount_mean', 'discount_max', 'shipping_fee_mean', 'shipping_days_mean'],
            "고객 문의": ['question_count', 'answer_rate']
        }

        for group_name, features in feature_groups.items():
            with st.expander(group_name, expanded=True):
                group_data = []
                for feat in features:
                    value = seller_features[feat]
                    # 값 포맷팅
                    if 'rate' in feat or 'discount' in feat:
                        formatted_value = f"{value:.1%}" if value <= 1 else f"{value:.1f}%"
                    elif 'price' in feat or 'fee' in feat:
                        formatted_value = f"₩{value:,.0f}"
                    elif isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)

                    group_data.append({
                        '피처': FEATURE_NAMES_KR.get(feat, feat),
                        '값': formatted_value
                    })

                st.table(pd.DataFrame(group_data))

        # 피처 중요도 (모델이 Random Forest인 경우)
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.subheader("피처 중요도 (모델 기준)")

            importance_df = pd.DataFrame({
                'feature': FEATURE_COLUMNS,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            fig_importance = go.Figure(data=go.Bar(
                x=importance_df['importance'],
                y=[FEATURE_NAMES_KR.get(f, f) for f in importance_df['feature']],
                orientation='h',
                marker_color='#636EFA'
            ))
            fig_importance.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title='중요도'
            )
            st.plotly_chart(fig_importance, use_container_width=True)


if __name__ == "__main__":
    main()
