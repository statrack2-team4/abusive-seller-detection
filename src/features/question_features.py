from src.features.text_utils import clean_text
import pandas as pd




QUESTION_KEYWORDS = {
    "refund": ["환불", "반품", "취소"],
    "authenticity": ["정품", "가짜", "짝퉁"],
    "defect": ["불량", "고장", "문제", "작동안됨"],
    "delivery": ["배송", "언제", "도착"],
    "usage": ["사용법", "먹는법", "어떻게"]
}


def classify_question(text: str):
    for k, keywords in QUESTION_KEYWORDS.items():
        if any(word in text for word in keywords):
            return k
    return "other"


def build_question_features(questions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    questions = questions.copy()
    products = products.copy()

    prod_vendor = products[["product_id", "vendor_name"]]
    questions = questions.merge(prod_vendor, on="product_id", how="left")

    questions["clean_text"] = questions["question"].apply(clean_text)
    questions["type"] = questions["clean_text"].apply(classify_question)

    result = []

    for vendor, g in questions.groupby("vendor_name"):
        total = len(g)

        feature = {
            "vendor_name": vendor,
            "total_question_count": total,
            "refund_question_ratio": (g["type"] == "refund").mean(),
            "authenticity_question_ratio": (g["type"] == "authenticity").mean(),
            "defect_question_ratio": (g["type"] == "defect").mean(),
            "delivery_question_ratio": (g["type"] == "delivery").mean(),
        }

        result.append(feature)

    return pd.DataFrame(result)
