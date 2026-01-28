from src.features.text_utils import (
    clean_text,
    text_length,
    negative_keyword_ratio,
    duplicate_ratio,
)
import pandas as pd


NEGATIVE_KEYWORDS = [
    "불량",
    "고장",
    "환불",
    "가짜",
    "사기",
    "효과없음",
    "부작용",
    "작동안됨",
    "충전안됨",
    "트러블",
    "피부염",
]


def build_review_features(
    reviews: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    """
    리뷰 기반 판매자 단위 feature 생성
    """

    reviews = reviews.copy()
    products = products.copy()

    # product_id → vendor_name 매핑
    prod_vendor = products[["product_id", "vendor_name"]]
    reviews = reviews.merge(prod_vendor, on="product_id", how="left")

    reviews["clean_text"] = reviews["review_text"].apply(clean_text)
    reviews["text_length"] = reviews["clean_text"].apply(text_length)

    result = []

    for vendor, g in reviews.groupby("vendor_name"):
        texts = g["clean_text"].tolist()
        ratings = g["review_rating"]

        feature = {
            "vendor_name": vendor,
            "avg_rating": ratings.mean(),
            "rating_std": ratings.std(),
            "low_rating_ratio": (ratings <= 2).mean(),
            "avg_review_length": g["text_length"].mean(),
            "negative_keyword_ratio": negative_keyword_ratio(texts, NEGATIVE_KEYWORDS),
            "duplicate_review_ratio": duplicate_ratio(texts),
            "review_count": len(g),
        }

        result.append(feature)

    return pd.DataFrame(result)
