# Features

Здесь лежат Python-модули для стабильной версии обработки признаков.

## Как использовать
1. Сначала исследуйте логику в ноутбуке группы.
2. Затем перенесите ее в `feature_processor.py` своей группы.
3. При необходимости дополните `example_feature_spec.json`.
4. Запускайте сборку через `src/assembler/assemble_dataset.py`.

## Структура по группам
Для каждой группы:
- `feature_processor.py` — основной код обработки
- `example_feature_spec.json` — необязательные внешние описания признаков
- `example_usage.py` — минимальный пример запуска

## Импорты
В проекте используются импорты от корня проекта, например:
```python
from src.utils.feature_spec import FeatureSpec
from src.utils.io import load_feature_names_from_txt
```

Запускайте скрипты из корня проекта.
