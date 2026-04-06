from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_feature_names_from_txt(path: str | Path) -> List[str]:
    """
    Загружает список признаков из txt-файла.

    Ожидаемый формат:
    - пустые строки игнорируются
    - строки вида '1. feature_name' поддерживаются
    - строки без нумерации тоже поддерживаются
    """
    path = Path(path)
    feature_names: List[str] = []

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            # Пропускаем заголовочные строки вроде "Группа признаков 1"
            lower = line.lower()
            if lower.startswith("группа признаков"):
                continue
            if set(line) == {"="}:
                continue

            # Убираем нумерацию "1. feature_name"
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
    2. словарь вида {"features": [...]}
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and "features" in payload:
        features = payload["features"]
        if isinstance(features, list):
            return features

    raise ValueError("JSON должен быть либо списком, либо словарем с ключом 'features'.")
