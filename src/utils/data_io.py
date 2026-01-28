import os
from pathlib import Path
import pandas as pd


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """
    저장 경로의 디렉토리가 존재하지 않으면 생성하고,
    DataFrame을 CSV 파일로 저장합니다. (encoding='utf-8-sig')

    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        output_path (Path): 저장할 파일 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 데이터 저장 완료: {output_path}")
