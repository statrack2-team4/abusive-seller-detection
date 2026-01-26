# Data Schemas

## 1. 상품

| 필드명 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- |
| `name` | `str` | - | 상품명 |
| `product_id` | `str` | - | 상품ID |
| `item_code` | `str` | - | 상품아이템코드 |
| `review_count` | `int` | - | 리뷰 수 |
| `is_seller_rocket` | `bool` | - | 판매자 로켓 여부 |
| `product_rating` | `float` | `0.0` | 상품 전체 평점 |
| `vendor_name` | `str` | - | 판매자 상호 (ID를 찾을 수 없는 경우가 있어 이름으로 사용) |
| `category` | `str` | `""` | 카테고리 |
| `price` | `int` | - | 가격 |
| `original_price` | `int` | - | 정가 |
| `candidate_price` | `int` | - | 후보가격 |
| `discount_rate` | `int` | - | 할인율 |
| `shipping_fee` | `str` | `""` | 배송비 |
| `shipping_days` | `str` | `""` | 배송소요일 |

## 2. Q&A

상품 Q&A 정보를 담고 있는 스키마입니다.

| 필드명 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- |
| `product_id` | `str` | - | 상품 ID |
| `question` | `str` | - | 질문 내용 |
| `question_date` | `str` | - | 질문 날짜 |
| `answer` | `str` | `""` | 답변 내용 |
| `answer_date` | `str` | `""` | 답변 날짜 |

## 3. 리뷰

| 필드명 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- |
| `product_id` | `str` | - | 상품 ID |
| `review_date` | `str` | - | 리뷰 날짜 |
| `review_title` | `str` | - | 리뷰 제목 |
| `review_text` | `str` | - | 리뷰 내용 |
| `review_rating` | `int` | - | 리뷰 평점 |

## 4. 판매자

| 필드명 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- |
| `company_name` | `str` | `""` | 상호 |
| `representative_name` | `str` | `""` | 대표자 |
| `business_registration_number` | `str` | `""` | 사업자번호 |
| `business_address` | `str` | `""` | 사업장 소재지 |
| `email` | `str` | `""` | 이메일 |
| `phone` | `str` | `""` | 연락처 |
| `communication_sales_number` | `str` | `""` | 통신판매업 신고번호 |
| `seller_id` | `Optional[str]` | `None` | 판매자ID |
| `satisfaction_score` | `int` | `0` | 만족도 |
| `review_count` | `int` | `0` | 리뷰 수 |
| `total_product_count` | `int` | `0` | 상품 수 |
