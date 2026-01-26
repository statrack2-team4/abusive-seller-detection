import os
from functools import lru_cache
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Supabase 클라이언트를 생성하고 반환합니다.

    환경 변수:
        SUPABASE_URL: Supabase 프로젝트 URL
        SUPABASE_KEY: Supabase anon/service role key

    Returns:
        Supabase Client 인스턴스

    Raises:
        ValueError: 환경 변수가 설정되지 않은 경우
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            """
            .env에 SUPABASE_URL과 SUPABASE_KEY 환경 변수를 설정해주세요.
            예시:
            SUPABASE_URL='https://your-project.supabase.co'
            SUPABASE_KEY='your-anon-key'
            """
        )

    return create_client(url, key)
