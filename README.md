
# Project: notebook-first feature engineering workflow

## Как работать

### Вариант 1. Исследование через ноутбуки

1. Откройте ноутбук:
   notebooks/group_X/feature_processing_template.ipynb

2. Проверьте пути:
   - DATA_PATH = "../../data/raw/MIPT_hackathon_dataset.csv"
   - FEATURES_PATH = "../../data/feature_groups/features_group_X.txt"
   - OUTPUT_DIR = "../../notebook_outputs/group_X"

3. Сделайте обработку признаков

4. Сохраните:
   - X_block.csv
   - feature_spec.csv

5. Соберите датасет:
   python src/assembler/assemble_dataset_notebook.py

---
                                                          
- CSV = рабочий документ аналитика
- JSON = контракт для кода

---

## 

1. Notebook:
   - строим X_block
   - создаем только список признаков:

   feature_spec = pd.DataFrame({"name": X_block.columns})

2. Ручное заполнение CSV:
   - source
   - group
   - description
   - baseline
   - leakage_risk
   - owner

3. Конвертация CSV → JSON:

   from src.utils.spec_converter import convert_feature_spec_csv_to_json

   convert_feature_spec_csv_to_json(
       "notebook_outputs/group_5/feature_spec.csv",
       "src/features/group_5/example_feature_spec.json"
   )

4. Код использует только JSON

## Вариант 2. Сборка из Python-модулей

1. Перенести логику из ноутбука в feature_processor.py
2. Использовать example_feature_spec.json
3. Собрать через assemble_dataset()

---

## Важные замечания

- CSV редактируется вручную
- JSON генерируется автоматически
- assemble_dataset_notebook читает только notebook_outputs

---

## Примеры обработки

group_5:
- lead_Ширина
- lead_Линейная высота (см)
- lead_Вид оплаты
- returned_ts
- lead_Служба доставки

---

## Коротко

notebook → CSV → JSON → code
