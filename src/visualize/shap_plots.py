"""
SHAP 시각화 모듈

SHAP 분석 결과를 Plotly로 시각화하는 함수들을 제공합니다.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_shap_waterfall(
    shap_values,
    sample_idx: int,
    feature_columns: list,
    feature_values,
    base_value: float,
    top_n: int = 20,
    title: str = None,
    height: int = 600,
) -> go.Figure:
    """
    SHAP Waterfall 차트를 Plotly로 시각화합니다.

    개별 샘플의 예측에 각 피처가 어떻게 기여했는지 보여주는 waterfall 차트입니다.
    Base value에서 시작하여 각 피처의 SHAP 값만큼 누적되어 최종 예측값에 도달합니다.

    Args:
        shap_values: 해당 샘플의 SHAP 값 (1D array, shape: n_features)
        sample_idx: 샘플 인덱스 (제목 표시용)
        feature_columns: 피처 이름 리스트
        feature_values: 해당 샘플의 실제 피처값 (pd.Series 또는 array)
        base_value: SHAP base value (평균 예측값)
        top_n: 표시할 상위 피처 수 (기본값: 20)
        title: 차트 제목 (None이면 자동 생성)
        height: 차트 높이 (기본값: 600)

    Returns:
        go.Figure: Plotly Figure 객체

    Example:
        >>> import shap
        >>> explainer = shap.TreeExplainer(model)
        >>> shap_values = explainer.shap_values(X_test)
        >>>
        >>> # Class 1(양성)에 대한 SHAP 값 추출
        >>> sv_class1 = shap_values[sample_idx][:, 1]
        >>> base_value = explainer.expected_value[1]
        >>>
        >>> fig = plot_shap_waterfall(
        ...     shap_values=sv_class1,
        ...     sample_idx=0,
        ...     feature_columns=feature_columns,
        ...     feature_values=X_test.iloc[0],
        ...     base_value=base_value,
        ...     top_n=15
        ... )
        >>> fig.show()
    """
    # feature_values 처리
    if hasattr(feature_values, "values"):
        feature_vals = feature_values.values
    else:
        feature_vals = feature_values

    # 데이터프레임 생성
    df_shap = pd.DataFrame(
        {
            "feature": feature_columns,
            "shap_value": shap_values,
            "feature_value": feature_vals,
        }
    )

    # SHAP 값의 절대값 크기순으로 정렬 (상위 top_n개만)
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    df_shap = df_shap.sort_values("abs_shap", ascending=True).tail(top_n)

    # Waterfall 차트 생성
    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(df_shap),
            y=df_shap["feature"],
            x=df_shap["shap_value"],
            text=df_shap["feature_value"].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else str(x)
            ),
            textposition="outside",
            connector={
                "mode": "between",
                "line": {"width": 1, "color": "rgb(150,150,150)", "dash": "dot"},
            },
        )
    )

    # 최종 예측값 계산
    total_shap = shap_values.sum() if hasattr(shap_values, "sum") else sum(shap_values)
    final_prediction = base_value + total_shap

    # 제목 설정
    if title is None:
        title = (
            f"<b>Sample #{sample_idx} 예측 설명 (Waterfall)</b><br>"
            f"Base: {base_value:.3f} → Prediction: {final_prediction:.3f}"
        )

    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=False,
        height=height,
        xaxis=dict(title="SHAP Value (기여도)"),
        template="plotly_white",
    )

    return fig


def plot_shap_summary(
    shap_values,
    feature_columns: list,
    title: str = "SHAP 피처 중요도",
    height: int = 600,
) -> go.Figure:
    """
    SHAP 피처 중요도 막대 차트를 Plotly로 시각화합니다.

    Args:
        shap_values: SHAP 값 (2D array, shape: n_samples x n_features)
        feature_columns: 피처 이름 리스트
        title: 차트 제목
        height: 차트 높이

    Returns:
        go.Figure: Plotly Figure 객체
    """
    # 평균 절대 SHAP 값 계산
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": mean_abs_shap}
    ).sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="mean(|SHAP value|)",
        yaxis_title="Feature",
        height=height,
        template="plotly_white",
    )

    return fig
