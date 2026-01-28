import os
from functools import lru_cache
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()


def _get_credentials():
    """환경 변수 또는 Streamlit Secrets에서 자격 증명을 가져옵니다."""
    # 1. 환경 변수 확인 (.env 또는 시스템 환경 변수)
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if url and key:
        return url, key

    # 2. Streamlit Secrets 확인 (Streamlit Cloud 배포용)
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        if url and key:
            return url, key
    except (ImportError, AttributeError, KeyError):
        pass

    return None, None


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Supabase 클라이언트를 생성하고 반환합니다.

    환경 변수 (우선순위):
        1. .env 파일 또는 시스템 환경 변수
        2. Streamlit Secrets (Streamlit Cloud 배포 시)

    Returns:
        Supabase Client 인스턴스

    Raises:
        ValueError: 자격 증명이 설정되지 않은 경우
    """
    url, key = _get_credentials()

    if not url or not key:
        raise ValueError(
            """
            Supabase 자격 증명을 설정해주세요.

            [로컬 개발]
            .env 파일에 다음을 추가:
            SUPABASE_URL='https://your-project.supabase.co'
            SUPABASE_KEY='your-anon-key'

            [Streamlit Cloud]
            앱 설정 → Secrets에 다음을 추가:
            SUPABASE_URL = "https://your-project.supabase.co"
            SUPABASE_KEY = "your-anon-key"
            """
        )

    return create_client(url, key)
