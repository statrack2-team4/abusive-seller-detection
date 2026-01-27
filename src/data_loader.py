"""
데이터 로드 모듈 - Supabase에서 데이터 로드 및 검증
"""

import pandas as pd
from src.database.supabase_client import get_supabase_client


class DataLoadError(Exception):
    """데이터 로드 관련 에러"""
    pass


def load_table(table_name: str, validate=True):
    """
    Supabase에서 테이블 데이터 로드
    
    Args:
        table_name: 테이블명 (products, reviews, questions, sellers)
        validate: 데이터 검증 여부
    
    Returns:
        pandas DataFrame
    
    Raises:
        DataLoadError: 데이터 로드 실패 시
    """
    try:
        supabase = get_supabase_client()
    except Exception as e:
        raise DataLoadError(f"Supabase 클라이언트 초기화 실패: {e}")
    
    try:
        print(f"  Loading {table_name}...", end=" ")
        res = supabase.table(table_name).select("*").execute()
        df = pd.DataFrame(res.data)
        
        if validate:
            _validate_dataframe(df, table_name)
        
        print(f"✅ {len(df)}행")
        return df
        
    except Exception as e:
        print(f"❌ 실패")
        raise DataLoadError(f"{table_name} 로드 중 에러 발생: {str(e)}")


def _validate_dataframe(df: pd.DataFrame, table_name: str):
    """
    데이터프레임 기본 검증
    
    Args:
        df: 검증할 데이터프레임
        table_name: 테이블명
    
    Raises:
        DataLoadError: 검증 실패 시
    """
    # 빈 데이터프레임 확인
    if df is None or len(df) == 0:
        raise DataLoadError(f"{table_name} 테이블이 비어있습니다.")
    
    # 필수 컬럼 확인 (스키마 기준)
    required_columns = {
        "products": ["product_id", "vendor_name"],  # products는 vendor_name
        "reviews": ["product_id", "review_text", "review_rating"],
        "questions": ["product_id", "question"],
        "sellers": ["company_name"]  # sellers는 company_name
    }
    
    if table_name in required_columns:
        missing = set(required_columns[table_name]) - set(df.columns)
        if missing:
            raise DataLoadError(
                f"{table_name} 테이블에 필수 컬럼이 없습니다: {missing}"
            )


def load_all_data(validate=True):
    """
    모든 테이블 데이터 로드
    
    Args:
        validate: 데이터 검증 여부
    
    Returns:
        dict: {"products": df, "reviews": df, "questions": df, "sellers": df}
    
    Raises:
        DataLoadError: 데이터 로드 실패 시
    """
    print("\n=== 데이터 로드 시작 ===")
    
    try:
        data = {
            "products": load_table("products", validate),
            "reviews": load_table("reviews", validate),
            "questions": load_table("questions", validate),
            "sellers": load_table("sellers", validate)
        }
        
        print("=== 데이터 로드 완료 ===\n")
        
        # 로드 후 추가 검증
        if validate:
            _validate_data_relationships(data)
        
        return data
        
    except Exception as e:
        print(f"\n❌ 데이터 로드 실패: {str(e)}")
        raise


def _validate_data_relationships(data: dict):
    """
    테이블 간 관계 검증
    
    Args:
        data: load_all_data 결과
    """
    print("데이터 관계 검증 중...", end=" ")
    
    products = data["products"]
    reviews = data["reviews"]
    questions = data["questions"]
    
    # 리뷰의 product_id가 products에 존재하는지
    review_products = set(reviews["product_id"].unique())
    valid_products = set(products["product_id"].unique())
    
    orphan_reviews = review_products - valid_products
    if orphan_reviews:
        print(f"\n  ⚠️ 경고: {len(orphan_reviews)}개 상품의 리뷰가 products 테이블에 없습니다.")
    
    # 문의의 product_id가 products에 존재하는지
    question_products = set(questions["product_id"].unique())
    orphan_questions = question_products - valid_products
    
    if orphan_questions:
        print(f"\n  ⚠️ 경고: {len(orphan_questions)}개 상품의 문의가 products 테이블에 없습니다.")
    
    if not orphan_reviews and not orphan_questions:
        print("✅")


def get_data_summary(data: dict):
    """
    로드된 데이터 요약 정보 출력
    
    Args:
        data: load_all_data 결과
    """
    print("\n=== 데이터 요약 ===")
    
    products = data["products"]
    reviews = data["reviews"]
    questions = data["questions"]
    sellers = data["sellers"]
    
    print(f"상품: {len(products)}개")
    print(f"  - 판매자: {products['vendor_name'].nunique()}명")
    print(f"  - 카테고리: {products['category'].nunique()}개" if "category" in products.columns else "")
    
    print(f"\n리뷰: {len(reviews)}개")
    print(f"  - 평균 평점: {reviews['review_rating'].mean():.2f}")
    print(f"  - 텍스트 없는 리뷰: {reviews['review_text'].isnull().sum()}개")
    
    print(f"\n문의: {len(questions)}개")
    print(f"  - 텍스트 없는 문의: {questions['question'].isnull().sum()}개")
    
    print(f"\n판매자: {len(sellers)}개")
    
    # 판매자당 평균 상품 수
    products_per_seller = products.groupby("vendor_name").size()
    print(f"  - 판매자당 평균 상품 수: {products_per_seller.mean():.1f}개")
    
    # 상품당 평균 리뷰 수
    reviews_per_product = reviews.groupby("product_id").size()
    print(f"  - 상품당 평균 리뷰 수: {reviews_per_product.mean():.1f}개")
    
    # 상품당 평균 문의 수
    questions_per_product = questions.groupby("product_id").size()
    print(f"  - 상품당 평균 문의 수: {questions_per_product.mean():.1f}개")
    
    print()


# 하위 호환성을 위한 별칭
def load_all():
    """
    레거시 함수 (하위 호환성)
    
    Returns:
        tuple: (products, reviews, questions, sellers)
    """
    data = load_all_data()
    return data["products"], data["reviews"], data["questions"], data["sellers"]


if __name__ == "__main__":
    # 테스트 실행
    try:
        data = load_all_data(validate=True)
        get_data_summary(data)
    except DataLoadError as e:
        print(f"에러: {e}")