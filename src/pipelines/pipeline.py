"""
sklearn Pipeline 기반 어뷰징 판매자 탐지 모델

사용 예시:
    from src.models import AbusingDetectorPipeline

    # 파이프라인 생성 및 학습
    pipeline = AbusingDetectorPipeline()
    pipeline.fit(X_train, y_train)

    # 예측
    predictions = pipeline.predict(X_test)

    # 저장 및 로드
    pipeline.save("models/pipeline.pkl")
    loaded = AbusingDetectorPipeline.load("models/pipeline.pkl")
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# =============================================================================
# 커스텀 Transformer 클래스
# =============================================================================


class FeatureSelector(BaseEstimator, TransformerMixin):
    """지정된 피처만 선택하는 Transformer"""

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.feature_names is None:
            return X
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        return X[:, : len(self.feature_names)]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


class OutlierClipper(BaseEstimator, TransformerMixin):
    """이상치를 클리핑하는 Transformer (IQR 기반)"""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.lower_bounds_ = np.percentile(X, self.lower_quantile * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_clipped = X.copy()
            for i, col in enumerate(X.columns):
                X_clipped[col] = np.clip(
                    X_clipped[col], self.lower_bounds_[i], self.upper_bounds_[i]
                )
            return X_clipped
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class TreeBasedFeatureSelector(BaseEstimator, TransformerMixin):
    """트리 기반 모델로 중요 피처만 선택하는 Transformer"""

    def __init__(
        self,
        threshold: str = "mean",
        estimator: Optional[BaseEstimator] = None,
    ):
        self.threshold = threshold
        self.estimator = estimator
        self.selector_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X = X.values

        if self.estimator is None:
            self.estimator = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

        self.estimator.fit(X, y)
        self.selector_ = SelectFromModel(
            self.estimator, threshold=self.threshold, prefit=True
        )
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is not None:
            mask = self.get_support()
            return [f for f, m in zip(self.feature_names_in_, mask) if m]
        return None


# =============================================================================
# 메인 파이프라인 클래스
# =============================================================================


class AbusingDetectorPipeline:
    """
    어뷰징 판매자 탐지를 위한 통합 ML 파이프라인

    구성:
        1. 피처 선택 (선택적)
        2. 이상치 클리핑 (선택적)
        3. 스케일링 (선택적)
        4. 피처 중요도 기반 선택 (선택적)
        5. 분류 모델

    Args:
        model_type: 모델 유형 ("rf", "gb", "lr", "voting")
        use_scaler: 스케일링 사용 여부
        use_feature_selection: 피처 선택 사용 여부
        use_outlier_clip: 이상치 클리핑 사용 여부
        feature_names: 사용할 피처 목록
        random_state: 랜덤 시드
    """

    # 기본 피처 목록 (legacy features 호환)
    DEFAULT_FEATURES = [
        "satisfaction_score",
        "review_count",
        "total_product_count",
        "product_count_actual",
        "price_mean",
        "price_std",
        "price_min",
        "price_max",
        "rating_mean",
        "rating_std",
        "review_sum",
        "review_mean",
        "discount_mean",
        "discount_max",
        "shipping_fee_mean",
        "shipping_days_mean",
        "review_count_actual",
        "review_rating_mean",
        "review_rating_std",
        "review_length_mean",
        "review_length_std",
        "review_length_max",
        "question_count",
        "answer_rate",
    ]

    def __init__(
        self,
        model_type: str = "rf",
        use_scaler: bool = True,
        use_feature_selection: bool = False,
        use_outlier_clip: bool = False,
        feature_names: list[str] | None = None,
        random_state: int = 42,
        **model_params,
    ):
        self.model_type = model_type
        self.use_scaler = use_scaler
        self.use_feature_selection = use_feature_selection
        self.use_outlier_clip = use_outlier_clip
        self.feature_names = feature_names or self.DEFAULT_FEATURES
        self.random_state = random_state
        self.model_params = model_params

        self.pipeline_ = None
        self.is_fitted_ = False
        self.selected_features_ = None

    def _create_model(self) -> BaseEstimator:
        """모델 인스턴스 생성"""
        if self.model_type == "rf":
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            return RandomForestClassifier(**default_params)

        elif self.model_type == "gb":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "subsample": 0.8,
                "random_state": self.random_state,
                "validation_fraction": 0.15,
                "n_iter_no_change": 10,
            }
            default_params.update(self.model_params)
            return GradientBoostingClassifier(**default_params)

        elif self.model_type == "lr":
            default_params = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "saga",
                "class_weight": "balanced",
                "random_state": self.random_state,
                "max_iter": 2000,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            return LogisticRegression(**default_params)

        elif self.model_type == "voting":
            return VotingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=100,
                            max_depth=5,
                            class_weight="balanced",
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    ),
                    (
                        "gb",
                        GradientBoostingClassifier(
                            n_estimators=100,
                            max_depth=3,
                            random_state=self.random_state,
                        ),
                    ),
                    (
                        "lr",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            random_state=self.random_state,
                            max_iter=2000,
                        ),
                    ),
                ],
                voting="soft",
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _build_pipeline(self) -> Pipeline:
        """sklearn Pipeline 구성"""
        steps = []

        # 1. 피처 선택
        steps.append(("feature_selector", FeatureSelector(self.feature_names)))

        # 2. 이상치 클리핑 (선택적)
        if self.use_outlier_clip:
            steps.append(("outlier_clipper", OutlierClipper()))

        # 3. 스케일링 (선택적, LR/SVM 등에 필요)
        if self.use_scaler:
            # 트리 기반 모델은 스케일링 불필요하지만 Voting에 LR 포함시 필요
            if self.model_type in ["lr", "voting"]:
                steps.append(("scaler", StandardScaler()))

        # 4. 피처 중요도 기반 선택 (선택적)
        if self.use_feature_selection:
            steps.append(("feature_importance_selector", TreeBasedFeatureSelector()))

        # 5. 분류 모델
        steps.append(("classifier", self._create_model()))

        return Pipeline(steps)

    def fit(self, X, y):
        """파이프라인 학습"""
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True

        # 선택된 피처 저장
        if self.use_feature_selection:
            selector = self.pipeline_.named_steps.get("feature_importance_selector")
            if selector:
                self.selected_features_ = selector.get_feature_names_out()

        return self

    def predict(self, X) -> np.ndarray:
        """예측"""
        if not self.is_fitted_:
            raise RuntimeError(
                "Pipeline이 학습되지 않았습니다. fit()을 먼저 호출하세요."
            )
        return self.pipeline_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted_:
            raise RuntimeError(
                "Pipeline이 학습되지 않았습니다. fit()을 먼저 호출하세요."
            )
        return self.pipeline_.predict_proba(X)

    def evaluate(self, X, y, verbose: bool = True) -> dict:
        """모델 평가"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
        }

        if verbose:
            print("=== 모델 평가 결과 ===")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
            print("\n" + classification_report(y, y_pred))

        return metrics

    def cross_validate(
        self, X, y, cv: int = 5, scoring: str = "f1", verbose: bool = True
    ) -> dict:
        """교차 검증"""
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # 새 파이프라인으로 교차검증
        pipeline = self._build_pipeline()
        scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring=scoring)

        results = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores,
        }

        if verbose:
            print(f"=== {cv}-Fold 교차검증 결과 ===")
            print(f"  {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")

        return results

    def get_feature_importance(self) -> pd.DataFrame | None:
        """피처 중요도 반환 (트리 기반 모델만)"""
        if not self.is_fitted_:
            return None

        classifier = self.pipeline_.named_steps["classifier"]

        # Voting의 경우 RF의 중요도 사용
        if isinstance(classifier, VotingClassifier):
            classifier = classifier.named_estimators_["rf"]

        if hasattr(classifier, "feature_importances_"):
            # 실제 사용된 피처 이름 가져오기
            if self.selected_features_:
                feature_names = self.selected_features_
            else:
                feature_names = self.feature_names

            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": classifier.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            return importance_df

        return None

    def save(self, filepath: str | Path):
        """파이프라인 저장"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "pipeline": self.pipeline_,
            "model_type": self.model_type,
            "use_scaler": self.use_scaler,
            "use_feature_selection": self.use_feature_selection,
            "use_outlier_clip": self.use_outlier_clip,
            "feature_names": self.feature_names,
            "selected_features": self.selected_features_,
            "random_state": self.random_state,
            "model_params": self.model_params,
        }

        joblib.dump(save_dict, filepath)
        print(f"파이프라인 저장 완료: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "AbusingDetectorPipeline":
        """저장된 파이프라인 로드"""
        save_dict = joblib.load(filepath)

        instance = cls(
            model_type=save_dict["model_type"],
            use_scaler=save_dict["use_scaler"],
            use_feature_selection=save_dict["use_feature_selection"],
            use_outlier_clip=save_dict["use_outlier_clip"],
            feature_names=save_dict["feature_names"],
            random_state=save_dict["random_state"],
            **save_dict["model_params"],
        )

        instance.pipeline_ = save_dict["pipeline"]
        instance.selected_features_ = save_dict["selected_features"]
        instance.is_fitted_ = True

        print(f"파이프라인 로드 완료: {filepath}")
        return instance


# =============================================================================
# 편의 함수
# =============================================================================


def create_pipeline(
    model_type: str = "rf",
    use_scaler: bool = True,
    use_feature_selection: bool = False,
    **kwargs,
) -> AbusingDetectorPipeline:
    """파이프라인 생성 편의 함수"""
    return AbusingDetectorPipeline(
        model_type=model_type,
        use_scaler=use_scaler,
        use_feature_selection=use_feature_selection,
        **kwargs,
    )


def load_pipeline(filepath: str | Path) -> AbusingDetectorPipeline:
    """파이프라인 로드 편의 함수"""
    return AbusingDetectorPipeline.load(filepath)


# =============================================================================
# CLI 실행
# =============================================================================


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.database import load_table
    from src.features.feature_generation import FeatureGenerator

    print("=== 어뷰징 탐지 Pipeline 테스트 ===\n")

    # 1. 데이터 로드
    print("[1] 데이터 로드 중...")
    generator = FeatureGenerator().load_data(from_db=True)
    features_df = generator.generate_legacy_features()

    X = features_df[AbusingDetectorPipeline.DEFAULT_FEATURES]
    y = features_df["is_abusing_seller"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. 파이프라인 학습
    print("\n[2] Pipeline 학습 중...")
    pipeline = AbusingDetectorPipeline(model_type="rf", use_feature_selection=False)
    pipeline.fit(X_train, y_train)

    # 3. 평가
    print("\n[3] 평가 결과:")
    metrics = pipeline.evaluate(X_test, y_test)

    # 4. 교차검증
    print("\n[4] 교차검증:")
    cv_results = pipeline.cross_validate(X_train, y_train, cv=5)

    # 5. 피처 중요도
    print("\n[5] 피처 중요도 (Top 10):")
    importance = pipeline.get_feature_importance()
    if importance is not None:
        print(importance.head(10).to_string(index=False))

    # 6. 저장
    print("\n[6] 파이프라인 저장:")
    pipeline.save("models/abusing_pipeline.pkl")
