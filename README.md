

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
- run_selector.py - позволяет на основании исходного предобработанного
dataset-а final_dataset_from_notebooks.csv создавать различные 
dataset-ы с разными наборами признаков пользуясь различными стратегиями.
Нужно для поиска самых важных признаков, удаления шума уменьшения числа
признаков без уменьшения предсказательной способности. 
Может служить ядром для развития полностью автоматизированной 
предобработки признаков с применением LLM.
- run_quick_model_check.py - быстрая проверка метрик сгенерированного 
dataset-а. Можно подключать разные модели для проверки.  
#todo скомбинировать работу по созданию dataset-ов и проверке качества 
полученных предсказаний, для поиска оптимума в рамках выбранных 
стратегий.
- скрипт data_preprocessing.py - создает новый датасет, на основании 
исходного датасета. Для этого использует преобразования
выбранных признаков. Этот список можно произвольно расширять, для этого
необходимо:
  - добавить новый метод transform_"New_Feature" в класс 
feature_audit.selector.manual_feature_extraction.ManualFeatureExtractor,
  - в этом же классе, добавить вызов нового метода в методе transform(),
  - и добавить название нового (признака или признаков) в фильтр 
    final_features_list в скрипте data_preprocessing.py
  
  Новый датасет появится в каталоге
  PROJECT_ROOT\data\cleaned_dataset\final_dataset.csv
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

