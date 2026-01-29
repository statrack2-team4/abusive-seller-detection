from src.database.supabase_client import get_supabase_client
import pandas as pd
from src.config import RAW_DATA_DIR


def load_table(table_name):
    # 속도, API 호출 횟수를 줄이기 위해 로컬에 저장된 데이터를 우선 확인
    local_path = RAW_DATA_DIR / f"{table_name}.csv"
    if local_path.exists():
        return pd.read_csv(local_path, encoding="utf-8")

    client = get_supabase_client()
    response = client.table(table_name).select("*").execute()

    # 로컬에 저장
    df = pd.DataFrame(response.data)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(df.to_csv(index=False), encoding="utf-8")
    return df
