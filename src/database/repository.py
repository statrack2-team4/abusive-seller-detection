from src.database.supabase_client import get_supabase_client
import pandas as pd


def load_table(table_name):
    client = get_supabase_client()
    response = client.table(table_name).select("*").execute()
    return pd.DataFrame(response.data)
