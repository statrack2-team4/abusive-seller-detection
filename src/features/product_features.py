import pandas as pd


def build_product_features(products_df: pd.DataFrame) -> pd.DataFrame:
    """판매자별 대표 상품 피처를 생성합니다."""
    # 판매자별 대표 상품 피처 (첫 번째 상품 기준)
    product_features = products_df.groupby("vendor_name").first().reset_index()

    # 평점 분포 비율 계산
    denom = product_features["review_count"].replace(0, 1)
    product_features["rating_5_ratio"] = product_features["rating_5"] / denom
    product_features["rating_4_ratio"] = product_features["rating_4"] / denom
    product_features["rating_1_2_ratio"] = (
        product_features["rating_1"] + product_features["rating_2"]
    ) / denom

    # 필요한 피처만 선택
    product_features = product_features[
        [
            "vendor_name",
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
        ]
    ].rename(columns={"vendor_name": "company_name"})

    return product_features
