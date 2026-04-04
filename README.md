
# Project: notebook-first feature engineering workflow

## Как работать

### Вариант 1. Исследование через ноутбуки

1. Откройте ноутбук:
   notebooks/group_X/feature_processing_template.ipynb
2. Проверьте пути:
   - DATA_PATH = "../../data/raw/MIPT_hackathon_dataset.csv"
   - FEATURES_PATH = "../../data/feature_groups/features_group_X.txt"
3. Сделайте анализ признаков
4. На основании анализа, модифицируйте текущий датасет
с целью очистки и преобразования данных. Для этого реализуйте логику 
очистки и преобразования в классе, наследуемом от BaseAnalyzer
feature_audit - analyser - base - BaseAnalyzer
5. Добавьте вызов своего класса в методе run feature_audit - 
   feature_cleaning_pipeline_base 
6. Запустите run_pipeline - будет сформирован новый очищенный датасет



