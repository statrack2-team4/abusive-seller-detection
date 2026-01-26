# 악성 판매자 탐지 (Abusive Seller Detection)

## 🎯 프로젝트 목표

오픈 마켓에서 악성 판매자를 탐지하는 머신러닝 프로젝트입니다.
"쿠팡"을 기준으로 데이터를 수집하고 악성 판매자 판별 모델을 학습시켰습니다.

## 😈 악성 판매자 판별

악성 판매자 판별은 휴리스틱한 방법으로 라벨러 한 명이 모든 판매자에 대해 직접 판단하고 라벨링했습니다.
쿠팡에서도 실시간으로 악성 판매자를 필터링하는 걸 확인할 수 있었기에 단순 가격, 평가, 배송 시간 비교등 룰베이스 기반은 이미 필터링 되고 있다고 가정했습니다.
그럼에도 사람이 보기에는 직관적으로 의심스러운 판매자들이 있었기에 휴리스틱한 방법으로 라벨링했습니다.

## 📦 준비

### 요구사항

* Python 3.13+
* [uv](https://github.com/astral-sh/uv) (패키지 관리자)

### 설치

```bash
uv sync
```

### 환경 변수 설정

Supabase를 사용하려면 환경 변수를 설정합니다:

```bash
export SUPABASE_URL='https://your-project.supabase.co'
export SUPABASE_KEY='your-service-role-key'
```
