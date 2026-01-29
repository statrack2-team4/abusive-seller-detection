from src.features.question_features import build_question_features
from src.features.review_features import build_review_features
from src.features.product_features import build_product_features
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_DATA_DIR


class FeatureGenerator:
    def __init__(
        self,
        data_dir: str | Path = PROCESSED_DATA_DIR,
        sellers_df=None,
        products_df=None,
        reviews_df=None,
        questions_df=None,
    ):
        self.data_dir = Path(data_dir)
        self.sellers_df = sellers_df
        self.products_df = products_df
        self.reviews_df = reviews_df
        self.questions_df = questions_df

    def load_data(self, from_db=False):
        """피처 엔지니어링에 필요한 데이터셋을 로드합니다."""
        if from_db:
            from src.database import load_table

            self.sellers_df = load_table("sellers")
            self.products_df = load_table("products")
            self.reviews_df = load_table("reviews")
            self.questions_df = load_table("questions")
        else:
            self.sellers_df = pd.read_csv(self.data_dir / "ml_sellers.csv")
            self.products_df = pd.read_csv(self.data_dir / "ml_products.csv")
            self.reviews_df = pd.read_csv(self.data_dir / "ml_reviews.csv")
            self.questions_df = pd.read_csv(self.data_dir / "ml_questions.csv")
        return self

    def generate_review_features(self) -> pd.DataFrame:
        """판매자별 리뷰 패턴 피처를 생성합니다."""
        # 판매자별 리뷰 매핑
        product_vendor_map = self.products_df[
            ["product_id", "vendor_name"]
        ].drop_duplicates()
        reviews_with_vendor = self.reviews_df.merge(
            product_vendor_map, on="product_id", how="left"
        )

        # 리뷰 텍스트 길이
        reviews_with_vendor["text_length"] = reviews_with_vendor["review_text"].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )

        # 짧은 리뷰 여부 (30자 미만)
        reviews_with_vendor["is_short_review"] = (
            reviews_with_vendor["text_length"] < 30
        ).astype(int)

        # 5점 리뷰 여부
        reviews_with_vendor["is_5_star"] = (
            reviews_with_vendor["review_rating"] == 5
        ).astype(int)

        # 판매자별 리뷰 패턴 통계
        review_pattern = (
            reviews_with_vendor.groupby("vendor_name")
            .agg(
                {
                    "id": "count",
                    "text_length": "mean",
                    "is_short_review": "mean",  # 짧은 리뷰 비율
                    "is_5_star": "mean",  # 5점 비율 (리뷰 기준)
                    "review_rating": "mean",
                }
            )
            .reset_index()
        )

        review_pattern.columns = [
            "company_name",
            "review_count_actual",
            "review_length_mean",
            "short_review_ratio",
            "five_star_ratio",
            "review_rating_mean",
        ]

        # 리뷰 유사도 계산
        review_similarity = (
            reviews_with_vendor.groupby("vendor_name")
            .apply(self._calculate_group_similarity, include_groups=False)
            .reset_index()
        )
        review_similarity.columns = ["company_name", "review_similarity"]

        return review_pattern.merge(review_similarity, on="company_name", how="left")

    @staticmethod
    def _calculate_group_similarity(group):
        """그룹 내 리뷰들 사이의 코사인 유사도를 계산하는 헬퍼 함수입니다."""
        texts = group["review_text"].dropna().tolist()
        if len(texts) < 2:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)

            # 대각선 제외한 평균 유사도
            n = sim_matrix.shape[0]
            total = sim_matrix.sum() - n  # 대각선(자기자신) 제외
            count = n * (n - 1)
            return total / count if count > 0 else 0.0
        except:
            return 0.0

    def generate_question_features(self) -> pd.DataFrame:
        """판매자별 문의 응대 피처를 생성합니다."""
        # 문의 데이터와 판매자 매핑
        question_vendor_map = self.products_df[
            ["product_id", "vendor_name"]
        ].drop_duplicates()
        questions_with_vendor = self.questions_df.merge(
            question_vendor_map, on="product_id", how="left"
        )

        # 날짜 변환
        questions_with_vendor["question_date"] = pd.to_datetime(
            questions_with_vendor["question_date"], errors="coerce"
        )
        questions_with_vendor["answer_date"] = pd.to_datetime(
            questions_with_vendor["answer_date"], errors="coerce"
        )

        # 답변 시간 계산 (시간 단위)
        questions_with_vendor["response_time_hours"] = (
            questions_with_vendor["answer_date"]
            - questions_with_vendor["question_date"]
        ).dt.total_seconds() / 3600

        # 답변 텍스트 길이
        questions_with_vendor["answer_length"] = questions_with_vendor["answer"].apply(
            lambda x: len(str(x)) if pd.notna(x) and x != "" else 0
        )

        # 짧은 답변 여부 (20자 미만)
        questions_with_vendor["is_short_answer"] = (
            questions_with_vendor["answer_length"] < 20
        ).astype(int)

        # 빠른 답변 여부 (24시간 이내)
        questions_with_vendor["is_quick_response"] = (
            (questions_with_vendor["response_time_hours"] > 0)
            & (questions_with_vendor["response_time_hours"] <= 24)
        ).astype(int)

        questions_with_vendor["is_answered"] = (
            questions_with_vendor["answer"].notna()
            & (questions_with_vendor["answer"] != "")
        ).astype(int)

        # 판매자별 문의 응대 패턴 집계
        question_features = (
            questions_with_vendor.groupby("vendor_name")
            .agg(
                {
                    "id": "count",  # 총 문의 수
                    "is_answered": "mean",  # 답변율
                    "response_time_hours": "mean",  # 평균 답변 시간
                    "is_quick_response": "mean",  # 빠른 답변 비율 (24시간 이내)
                    "is_short_answer": "mean",  # 짧은 답변 비율
                    "answer_length": "mean",  # 평균 답변 길이
                }
            )
            .reset_index()
        )

        question_features.columns = [
            "company_name",
            "question_count",  # 총 문의 수
            "answer_rate",  # 답변율 (0~1)
            "avg_response_hours",  # 평균 답변 시간 (시간)
            "quick_response_ratio",  # 24시간 이내 답변 비율
            "short_answer_ratio",  # 짧은 답변 비율
            "avg_answer_length",  # 평균 답변 길이
        ]

        # 답변이 없는 경우 response_time이 NaN이므로 처리
        question_features["avg_response_hours"] = question_features[
            "avg_response_hours"
        ].fillna(0)

        return question_features

    def generate_all_features(self, exclude_columns=[]) -> pd.DataFrame:
        """전체 피처 생성 파이프라인을 실행합니다."""
        if self.sellers_df is None:
            self.load_data()

        prod_feat_df = build_product_features(self.products_df)
        rev_feat_df = build_review_features(self.reviews_df, self.products_df)
        ques_type_feat_df = build_question_features(self.questions_df, self.products_df)
        ques_response_feat_df = self.generate_question_features()

        # 모든 피처 병합
        final_df = prod_feat_df.merge(rev_feat_df, on="company_name", how="outer")
        final_df = final_df.merge(ques_type_feat_df, on="company_name", how="outer")
        final_df = final_df.merge(ques_response_feat_df, on="company_name", how="outer")

        # 모든 판매자가 포함되도록 마스터 판매자 목록과 병합
        master_sellers = self.sellers_df[["company_name", "is_abusing_seller", "satisfaction_score"]]
        final_df = master_sellers.merge(final_df, on="company_name", how="left")

        # 상품/리뷰/문의가 없는 판매자를 위해 NaN 값을 처리합니다.
        # 일반적으로 카운트나 비율은 0으로, 또는 적절한 기본값으로 채웁니다.
        fill_zeros = [
            # product_features
            "price",
            "discount_rate",
            "product_rating",
            "shipping_fee",
            "shipping_days",
            "review_count",
            "inquiry_count",
            "rating_5_ratio",
            "rating_4_ratio",
            "rating_1_2_ratio",
            # review_features (build_review_features)
            "avg_rating",
            "rating_std",
            "low_rating_ratio",
            "avg_review_length",
            "negative_keyword_ratio",
            "duplicate_review_ratio",
            "review_count_actual",
            # question_features (build_question_features)
            "total_question_count",
            "refund_question_ratio",
            "authenticity_question_ratio",
            "defect_question_ratio",
            "delivery_question_ratio",
            # question_features (generate_question_features)
            "question_count",
            "answer_rate",
            "avg_response_hours",
            "quick_response_ratio",
            "short_answer_ratio",
            "avg_answer_length",
        ]
        # 존재하는 컬럼만 fillna 적용
        fill_zeros = [col for col in fill_zeros if col in final_df.columns]
        final_df[fill_zeros] = final_df[fill_zeros].fillna(0)

        final_df = final_df.drop(columns=exclude_columns)

        return final_df

    def generate_legacy_features(self) -> pd.DataFrame:
        """
        05_model_improvement.ipynb 및 07_error_analysis.ipynb에서 사용된
        세트와 일치하는 피처를 생성합니다.
        """
        if self.sellers_df is None:
            self.load_data()

        product_vendor_map = self.products_df[
            ["product_id", "vendor_name"]
        ].drop_duplicates()

        # 판매자별 상품 통계
        product_stats = (
            self.products_df.groupby("vendor_name")
            .agg(
                {
                    "product_id": "count",
                    "price": ["mean", "std", "min", "max"],
                    "product_rating": ["mean", "std"],
                    "review_count": ["sum", "mean"],
                    "discount_rate": ["mean", "max"],
                    "shipping_fee": "mean",
                    "shipping_days": "mean",
                }
            )
            .reset_index()
        )

        product_stats.columns = [
            "company_name",
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
        ]

        # 리뷰 통계
        reviews_with_vendor = self.reviews_df.merge(
            product_vendor_map, on="product_id", how="left"
        )
        reviews_with_vendor["text_length"] = reviews_with_vendor["review_text"].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )

        review_stats = (
            reviews_with_vendor.groupby("vendor_name")
            .agg(
                {
                    "id": "count",
                    "review_rating": ["mean", "std"],
                    "text_length": ["mean", "std", "max"],
                }
            )
            .reset_index()
        )

        review_stats.columns = [
            "company_name",
            "review_count_actual",
            "review_rating_mean",
            "review_rating_std",
            "review_length_mean",
            "review_length_std",
            "review_length_max",
        ]

        # 질문 통계
        questions_with_vendor = self.questions_df.merge(
            product_vendor_map, on="product_id", how="left"
        )
        questions_with_vendor["has_answer"] = questions_with_vendor["answer"].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip() != "" else 0
        )

        question_stats = (
            questions_with_vendor.groupby("vendor_name")
            .agg({"id": "count", "has_answer": "mean"})
            .reset_index()
        )
        question_stats.columns = ["company_name", "question_count", "answer_rate"]

        # 피처 병합
        features_df = self.sellers_df[
            [
                "company_name",
                "satisfaction_score",
                "review_count",
                "total_product_count",
                "is_abusing_seller",
            ]
        ].copy()

        features_df = features_df.merge(product_stats, on="company_name", how="left")
        features_df = features_df.merge(review_stats, on="company_name", how="left")
        features_df = features_df.merge(question_stats, on="company_name", how="left")
        features_df = features_df.fillna(0)

        return features_df


if __name__ == "__main__":
    generator = FeatureGenerator()
    features = generator.generate_all_features()
    print(f"Generated features for {len(features)} sellers.")
    print(features.head())
    # 필요한 경우 기본 위치에 저장
    # features.to_csv("data/processed/features.csv", index=False)
