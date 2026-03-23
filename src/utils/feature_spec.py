from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class FeatureSpec:
    """
    Краткое описание одного итогового признака.

    Поля:
    - name:
        Имя итоговой колонки в X_block / final_dataset.
    - source:
        Исходная raw-колонка или список raw-колонок.
    - group:
        Имя блока, например 'group_3'.
    - description:
        Короткое текстовое описание смысла и обработки признака.
    - baseline:
        True, если признак рекомендуется включать в baseline-набор.
    - leakage_risk:
        Риск утечки таргета: 'none', 'low', 'high'.
    """
    name: str
    source: str | List[str]
    group: str
    description: str
    baseline: bool
    leakage_risk: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
