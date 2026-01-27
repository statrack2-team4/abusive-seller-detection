from src.database.repository import load_table
from src.features.text_features import build_review_features
from src.features.question_features import build_question_features
import pandas as pd
import os

# 출력 폴더 생성
os.makedirs("output", exist_ok=True)

# 데이터 로드
products = load_table("products")
reviews = load_table("reviews")
questions = load_table("questions")
sellers = load_table("sellers")

print("✅ 데이터 로드 완료")
print(products.shape, reviews.shape, questions.shape, sellers.shape)

# feature 생성
review_feat = build_review_features(reviews, products)
question_feat = build_question_features(questions, products)

# 병합
df = sellers.merge(review_feat, left_on="company_name", right_on="vendor_name", how="left")
df = df.merge(question_feat, on="vendor_name", how="left")

# 기본 파생 변수
df["review_density"] = df["review_count_y"] / (df["total_product_count"] + 1)
df["question_to_review_ratio"] = df["total_question_count"] / (df["review_count_y"] + 1)

print("✅ Feature 생성 완료")
print(df.head())

# 저장
df.to_csv("output/seller_features.csv", index=False, encoding="utf-8-sig")
print("✅ output/seller_features.csv 저장 완료")
