import pandas as pd

def aggregate_all(products, sellers, review_feat, question_feat):
    df = sellers.merge(review_feat, left_on="company_name", right_on="vendor_name", how="left")
    df = df.merge(question_feat, left_on="company_name", right_on="vendor_name", how="left")

    df["review_density"] = df["total_reviews"] / df["total_product_count"]
    df["review_density"] = df["review_density"].fillna(0)

    df["negative_ratio"] = df["negative_kw_count"] / df["total_reviews"]
    df["negative_ratio"] = df["negative_ratio"].fillna(0)

    df = df.fillna(0)

    return df
