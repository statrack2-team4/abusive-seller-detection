from .supabase_client import get_supabase_client   # ← 여기만 변경
import pandas as pd


def load_table(table_name):
    client = get_supabase_client()
    response = client.table(table_name).select("*").execute()
    return pd.DataFrame(response.data)
