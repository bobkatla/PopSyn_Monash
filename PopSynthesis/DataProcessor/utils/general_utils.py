from pathlib import Path
from typing import Union


def find_file(base_path: str, filename: str) -> Union[None, Path]:
    base_path = Path(base_path)
    for file in base_path.rglob(filename):
        if file.is_file():
            return file
    return None
