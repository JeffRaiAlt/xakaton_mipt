

## Структура проекта


---

#### data/

Данные и описание признаков.

- MIPT_hackathon_dataset_справочник_полей.md — описание полей датасета  
- feature_groups/ — разбиение признаков на группы  
  - features_group_1.txt … features_group_5.txt — списки признаков по группам  

---

#### notebooks/

Рабочая зона для анализа и работы.
 
notebooks/group_2/ - model_processing_2.ipynb - основная рабочая модель


#### src/

Основной код проекта.

- run_pipeline.py — точка запуска пайплайна (пока только предварительная 
  очистка интегрирована)  
- report.py — формирование итоговых отчётов  

---

#### src/assembler/

Сборка финального датасета.

- assemble_dataset_notebook.py — сборка датасета  
- io.py — загрузка/сохранение данных  

---

#### src/feature_audit/

Анализ и очистка признаков.

- feature_cleaning_pipeline_base.py — базовый пайплайн очистки  
- logger.py — логирование этапов  
- utils.py — вспомогательные функции  

---

#### src/feature_audit/analyser/

Анализаторы признаков.

- base.py — базовый класс анализатора  
- cardinality_analyser.py — высокая кардинальность  
- categorical_target_correlation_analyser.py — связь категорий с target  
- date_analyser.py — анализ дат  
- date_candidates_analyser.py — кандидаты в даты  
- dominant_analyser.py — доминирующие значения  
- duplicates_analyser.py — дубликаты  
- empty_features_analyser.py — пустые признаки  
- feature_analyser_5_structured.py — структурный анализ  
- manual_dropper_analyser.py — ручное удаление  
- numeric_feature_correlation_analyser.py — корреляции числовых признаков  
- numeric_target_correlation_analyser.py — корреляция с target  
- order_analyser.py — временной порядок  
- tmp_feature_analyser_5.py — экспериментальный анализатор  

---

#### src/feature_audit/selector/

Отбор признаков на основе моделей и метрик.

- models/ — реализации моделей для отбора (CatBoost, RF, LogReg)  
- utils/ — вспомогательные функции для отбора  
- selector_pipeline.py — основной пайплайн отбора признаков  

