from pathlib import Path


def find_file(base_path, filename):
    base_path = Path(base_path)
    for file in base_path.rglob(filename):
        if file.is_file():
            return file
    return None
