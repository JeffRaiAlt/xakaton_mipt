# Notebooks

В этой папке лежат исследовательские ноутбуки по группам признаков.

## Назначение
Ноутбуки нужны для быстрой проверки гипотез:
- посмотреть распределения
- попробовать очистку и derived-признаки
- сформировать `X_block.csv`
- сформировать `feature_spec.csv`

После стабилизации логики ее нужно переносить в `src/features/group_X/feature_processor.py`.

## Где что лежит
- Ноутбук группы: `notebooks/group_X/feature_processing_template.ipynb`
- Список raw-признаков: `data/feature_groups/features_group_X.txt`
- Выход ноутбука: `notebook_outputs/group_X/`

## Важные пути внутри ноутбука
Ноутбук предполагает запуск из своей директории `notebooks/group_X/`, поэтому относительные пути должны быть такими:
- `../../data/raw/MIPT_hackathon_dataset.csv`
- `../../data/feature_groups/features_group_X.txt`
- `../../notebook_outputs/group_X`

## Формат результатов
Каждый ноутбук должен сохранить два файла:
- `X_block.csv`
- `feature_spec.csv`

Оба файла сохраняются в соответствующую папку `notebook_outputs/group_X/`.
