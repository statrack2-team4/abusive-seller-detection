from pathlib import Path
import os


def _get_project_root():
    # 현재 파일(또는 노트북)의 위치에서 상위로 올라가며 .git 폴더를 찾음
    current_path = Path(os.getcwd())  # 노트북 대응
    for parent in [current_path] + list(current_path.parents):
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return current_path  # 못 찾을 경우 현재 위치 반환


PROJECT_ROOT = _get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
