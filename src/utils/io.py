
import json
from pathlib import Path
from typing import Any, Dict, List


def load_feature_names_from_txt(path: str | Path) -> List[str]:
    """
    Загружает список признаков из txt-файла.

    Поддерживаемый формат:
    - пустые строки игнорируются
    - строка вида '1. feature_name' -> признак 'feature_name'
    - заголовки вроде 'Группа признаков 1' и строки из '=' игнорируются
    """
    path = Path(path)
    feature_names: List[str] = []

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith("группа признаков"):
                continue
            if set(line) == {"="}:
                continue

            if ". " in line:
                left, right = line.split(". ", 1)
                if left.isdigit():
                    line = right.strip()

            feature_names.append(line)

    return feature_names


def load_specs_from_json(path: str | Path) -> List[Dict[str, Any]]:
    """
    Загружает внешние описания признаков из JSON.

    Поддерживаются два формата:
    1. список словарей
    2. словарь вида {'features': [...]}
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and "features" in payload and isinstance(payload["features"], list):
        return payload["features"]

    raise ValueError("JSON должен быть либо списком, либо словарем с ключом 'features'.")
