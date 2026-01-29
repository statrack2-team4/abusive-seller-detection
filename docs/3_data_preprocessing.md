# 데이터 전처리 및 피처 엔지니어링

이 문서는 `notebooks/02_data_preprocessing.ipynb`와 `notebooks/03_feature_engineering.ipynb`에서 수행되는 데이터 전처리 및 피처 엔지니어링 파이프라인을 설명합니다.

## 1. 개요

원본 데이터(Raw Data)로부터 머신러닝 모델 학습에 적합한 형태의 데이터셋을 생성하는 과정을 다룹니다. 과정은 크게 **데이터 정제(Cleaning)** 단계와 **피처 생성(Feature Engineering)** 단계로 나뉩니다.

## 2. 데이터 정제 (Data Cleaning)

`notebooks/02_data_preprocessing.ipynb`에서 수행되며, 데이터베이스에서 로드한 `sellers`, `products`, `reviews`, `questions` 테이블을 정제하여 `data/processed/` 디렉토리에 `ml_*.csv` 형태로 저장합니다.

### 2.1 판매자 데이터 (Sellers)

- **주소 정제**: `business_address`에서 시/도 및 구/군 정보를 추출하여 `city_state` 컬럼 생성
- **불필요한 컬럼 제거**: 개인정보 및 모델링에 불필요한 컬럼 삭제 (`business_address`, `business_registration_number`, `email`, `phone`, `mail_order_number`, `created_at` 등)
- **해외 사업자 식별 (`is_cross_border`)**: `city_state`가 주요 국내 행정구역(서울, 경기 등)에 포함되지 않는 경우 1로 설정
- **영문 이름 식별 (`is_english_name`)**: 대표자명(`representative_name`)에 영문이 포함되어 있고, 해외 사업자로 식별된 경우 1로 설정

### 2.2 상품 데이터 (Products)

- **수치형 변환**: `price`, `original_price`, `review_count`, `product_rating`을 수치형으로 변환 및 결측치 0으로 채움
- **할인율 계산**: `discount_rate`가 없는 경우 `(original_price - price) / original_price`로 계산
- **카테고리 처리**: 결측값은 'Unknown'으로 대체

### 2.3 리뷰 및 문의 데이터 (Reviews & Questions)

- **날짜 변환**: 날짜 컬럼을 Datetime 객체로 변환
- **텍스트 정제**: 공백 제거 및 결측값 처리
- **답변 여부 태깅 (문의)**: `answer` 컬럼의 내용을 기반으로 `is_answered` 플래그 생성

---

## 3. 피처 엔지니어링 (Feature Engineering)

`notebooks/03_feature_engineering.ipynb`에서 수행되며, 정제된 데이터를 바탕으로 판매자별 특징을 집계하여 최종 `data/processed/features.csv`를 생성합니다.

### 3.1 상품 기반 피처
대부분의 판매자(약 92.8%)가 1개의 상품만 판매하므로, 판매자별 대표 상품(첫 번째 상품)의 정보를 사용합니다.

- **기본 정보**: 가격(`price`), 할인율(`discount_rate`), 평점(`product_rating`), 배송비(`shipping_fee`), 배송일(`shipping_days`), 리뷰 수(`review_count`), 문의 수(`inquiry_count`)
- **평점 분포 비율**:
  - `rating_5_ratio`: 5점 리뷰 비율
  - `rating_4_ratio`: 4점 리뷰 비율
  - `rating_1_2_ratio`: 1~2점(부정) 리뷰 비율

### 3.2 리뷰 패턴 피처
리뷰 데이터를 판매자 단위로 집계하여 비정상적인 리뷰 패턴을 탐지하기 위한 피처를 생성합니다.

- **`review_count_actual`**: 수집된 실제 리뷰 개수
- **`review_length_mean`**: 평균 리뷰 텍스트 길이
- **`short_review_ratio`**: 30자 미만 짧은 리뷰의 비율 (내용 없는 리뷰 탐지)
- **`five_star_ratio`**: 5점 만점 리뷰의 비율
- **`review_rating_mean`**: 평균 리뷰 평점

### 3.3 문의 응대 패턴 피처 (New)
악성 판매자의 불성실한 고객 응대 패턴(낮은 답변율, 느린 답변, 성의 없는 답변)을 탐지합니다.

- **`question_count`**: 총 접수된 문의 수
- **`answer_rate`**: 답변 완료율 (0~1)
  - 악성 판매자는 문의를 무시하는 경향이 있을 것으로 가정
- **`avg_response_hours`**: 평균 답변 소요 시간(시간 단위)
- **`quick_response_ratio`**: 24시간 이내 빠른 답변 비율
- **`short_answer_ratio`**: 20자 미만 짧은 답변 비율
- **`avg_answer_length`**: 평균 답변 텍스트 길이

### 3.4 리뷰 유사도 피처 (`review_similarity`)
조작된(어뷰징) 리뷰의 징후인 '복사해서 붙여넣기' 된 리뷰를 탐지합니다.

- **알고리즘**: TF-IDF (상위 100개 단어) -> 코사인 유사도(Cosine Similarity)
- **계산**: 같은 판매자의 리뷰들 간의 텍스트 유사도 평균값 (자기 자신과의 유사도 제외)
- **해석**: 값이 1에 가까울수록 모든 리뷰의 내용이 동일함(조작 의심)을 의미합니다.

## 4. 최종 데이터셋 (`features.csv`)

최종적으로 생성되는 데이터셋은 판매자(`company_name`)를 기준으로 하여 다음과 같은 컬럼을 가집니다.

| 분류 | 컬럼명 | 설명 |
|:---:|:---:|:---|
| **타겟/식별자** | `company_name` | 판매자명 |
| | `is_abusing_seller` | 어뷰징 판매자 여부 (Target) |
| | `satisfaction_score` | 만족도 점수 |
| **상품 정보** | `price` | 판매 가격 |
| | `discount_rate` | 할인율 |
| | `product_rating` | 상품 평점 |
| | `shipping_fee` | 배송비 |
| | `shipping_days` | 배송 소요 일수 |
| | `review_count` | 상품에 표시된 리뷰 수 |
| | `inquiry_count` | 문의 수 |
| **평점 분포** | `rating_5_ratio` | 5점 리뷰 비율 |
| | `rating_4_ratio` | 4점 리뷰 비율 |
| | `rating_1_2_ratio` | 1~2점 리뷰 비율 |
| **리뷰 패턴** | `review_count_actual` | 실제 수집된 리뷰 수 |
| | `review_length_mean` | 평균 리뷰 길이 |
| | `short_review_ratio` | 짧은 리뷰 비율 |
| | `five_star_ratio` | 수집된 리뷰 중 5점 비율 |
| | `review_rating_mean` | 수집된 리뷰의 평균 평점 |
| **문의/응대** | `question_count` | 판매자에 대한 총 문의 수 |
| | `answer_rate` | 답변율 |
| | `avg_response_hours` | 평균 답변 시간 |
| | `quick_response_ratio` | 빠른 답변 비율 |
| | `short_answer_ratio` | 짧은 답변 비율 |
| | `avg_answer_length` | 평균 답변 길이 |
| **심화 분석** | `review_similarity` | 리뷰 텍스트 유사도 |
