"""
모델을 Hugging Face Hub에 업로드하는 스크립트

사용법:
    1. Hugging Face 계정 생성: https://huggingface.co/join
    2. Access Token 생성: https://huggingface.co/settings/tokens (write 권한 필요)
    3. 환경 변수 설정 또는 로그인:
       - export HF_TOKEN=your_token
       - 또는 huggingface-cli login
    4. 스크립트 실행: python scripts/upload_model_to_hf.py
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# 설정
REPO_ID = "abusive-seller-detection"  # 변경 가능: "username/repo-name"
MODEL_PATH = Path("models/abusing_detector_tuned_tuned_rf.pkl")


def upload_model():
    api = HfApi()

    # 현재 사용자 정보 확인
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"로그인된 사용자: {username}")
    except Exception as e:
        print("Hugging Face에 로그인해주세요:")
        print("  1. huggingface-cli login")
        print("  2. 또는 export HF_TOKEN=your_token")
        raise e

    # 전체 repo_id 설정
    full_repo_id = f"{username}/{REPO_ID}"

    # 리포지토리 생성 (없으면)
    try:
        create_repo(full_repo_id, repo_type="model", exist_ok=True)
        print(f"리포지토리 준비 완료: {full_repo_id}")
    except Exception as e:
        print(f"리포지토리 생성 실패: {e}")
        raise

    # 모델 파일 업로드
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    print(f"모델 업로드 중: {MODEL_PATH}")
    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo=MODEL_PATH.name,
        repo_id=full_repo_id,
        repo_type="model",
    )

    print(f"\n✅ 업로드 완료!")
    print(f"   URL: https://huggingface.co/{full_repo_id}")
    print(f"\n다음 단계:")
    print(f"   app.py의 HF_REPO_ID를 '{full_repo_id}'로 설정하세요")


if __name__ == "__main__":
    upload_model()
