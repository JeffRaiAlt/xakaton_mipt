| Признак                             | Вердикт     | Пояснение                                  |
| ----------------------------------- | ----------- | ------------------------------------------ |
| contact_pvz_code                    | ⚠️ оставить | Может появляться после оформления          |
| lead_weight_gm                      | ✅ оставить  | Характеристика заказа                      |
| lead_responsible_user_id            | ✅ оставить  | Известен сразу                             |
| lead_utm_content                    | ✅ оставить  | Маркетинг                                  |
| contact_LTV                         | ❌ удалить   | Содержит будущие покупки                   |
| sale_date                           | ⚠️ оставить | Дубликат `sale_ts`, но удобен              |
| lead_utm_group                      | ✅ оставить  | Маркетинг                                  |
| lead_utm_referrer                   | ✅ оставить  | Маркетинг                                  |
| lead_tags                           | ⚠️ оставить | Может обновляться позже                    |
| lead_utm_source                     | ✅ оставить  | Маркетинг                                  |
| lead_Квалификация лида              | ⚠️ оставить | После звонка                               |
| is_responsible_same                 | ✅ оставить  | Производный, безопасный                    |
| has_weight                          | ✅ оставить  | Производный                                |
| has_contact_LTV                     | ❌ удалить   | Производный от утечки                      |
| contact_region_pvz                  | ⚠️ оставить | Может быть поздним                         |
| lead_utm_referrer_site              | ✅ оставить  | Маркетинг                                  |
| lead_has_roistat                    | ✅ оставить  | Маркетинг                                  |
| lead_utm_id_1/2/3                   | ✅ оставить  | Маркетинг                                  |
| lead_utm_device_type                | ✅ оставить  | Маркетинг                                  |
| lead_utm_site                       | ✅ оставить  | Маркетинг                                  |
| lead_utm_position                   | ✅ оставить  | Маркетинг                                  |
| lead_utm_reatrgeting_id             | ✅ оставить  | Маркетинг                                  |
| lead_utm_region_name                | ✅ оставить  | Маркетинг                                  |
| lead_is_utm_campaign_type_1         | ✅ оставить  | Маркетинг                                  |
| contact_Город                       | ✅ оставить  | География                                  |
| contact_region                      | ✅ оставить  | География                                  |
| lead_manager_category               | ⚠️ оставить | Может зависеть от процесса                 |
| lead_rate_is_warehouse_to_warehouse | ⚠️ оставить | Может зависеть от логистики после          |
| lead_formname_has_value             | ✅ оставить  | Маркетинг                                  |
| lead_has_creation_date              | ✅ оставить  | Технический                                |
| lead_creation_date_*                | ✅ оставить  | До `sale_ts`                               |
| sale_date_*                         | ✅ оставить  | Календарные признаки                       |
| buyout_flag_lag30                   | ❌ удалить   | Утечка (таргет в прошлом/агрегат)          |
| buyout_flag_lag60                   | ❌ удалить   | Утечка                                     |
| buyout_flag_ma30                    | ❌ удалить   | Утечка                                     |
| lead_items_count                    | ✅ оставить  | Состав заказа                              |
| lead_total_quantity                 | ✅ оставить  | Состав заказа                              |
| lead_total_cost_from_composition    | ⚠️ оставить | Проверить, не включает ли будущее          |
| lead_has_* (все категории)          | ✅ оставить  | Состав заказа                              |
| lead_categories_count               | ✅ оставить  | Состав заказа                              |
| lead_has_linear_width               | ✅ оставить  | Производный                                |
| lead_linear_width                   | ✅ оставить  | Характеристика                             |
| lead_discount_category              | ✅ оставить  | Цена                                       |
| lead_discount                       | ✅ оставить  | Цена                                       |
| sale_hour                           | ✅ оставить  | Время                                      |
| sale_dayofweek                      | ✅ оставить  | Время                                      |
| sale_month                          | ✅ оставить  | Время                                      |
| sale_ts                             | ⚠️ оставить | База                                       |
| lead_source                         | ✅ оставить  | Маркетинг                                  |
| timedelta_between_sale_and_creation | ✅ оставить  | История                                    |
| lead_created_ts                     | ✅ оставить  | История                                    |
| lead_created_dayofweek/hour/month   | ✅ оставить  | История                                    |
| lead_has_shipping_cost              | ✅ оставить  | Производный                                |
| lead_shipping_cost                  | ✅ оставить  | Цена доставки                              |
| lead_has_length                     | ✅ оставить  | Производный                                |
| lead_length                         | ✅ оставить  | Характеристика                             |
| lead_price                          | ✅ оставить  | Цена                                       |
| lead_pipeline_id                    | ⚠️ оставить | Может быть константой                      |
| lead_group_id                       | ⚠️ оставить | Проверить смысл                            |
| buyout_flag                         | ❌ удалить   | Target                                     |
| width_cat                           | ✅ оставить  | Производный                                |
| width_is_missing                    | ✅ оставить  | Производный                                |
| lead_payment_type                   | ✅ оставить  | Ключевой фактор                            |
| lead_delivery_type                  | ✅ оставить  | Ключевой фактор                            |
| lead_group_id_missing               | ⚠️ оставить | Производный                                |
| lead_group_quality                  | ⚠️ оставить | Может кодировать outcome                   |
| lead_mass_known                     | ✅ оставить  | Производный                                |
| lead_mass_log                       | ✅ оставить  | Производный                                |
| contact_to_lead_hours               | ⚠️ оставить | Проверить временную корректность           |
| contact_missing                     | ⚠️ оставить | Может коррелировать с процессом            |
| contact_hour/dayofweek/month        | ⚠️ оставить | Проверить момент записи                    |
| contact_is_weekend                  | ⚠️ оставить | То же                                      |
| is_paid_traffic                     | ✅ оставить  | Маркетинг                                  |
| lead_category_freq                  | ⚠️ оставить | Может считаться по всему датасету (утечка) |
| is_feature_phone                    | ✅ оставить  | Устройство                                 |
| lead_utm_campaign_missing           | ✅ оставить  | Производный                                |
| lead_utm_campaign_grouped           | ✅ оставить  | Маркетинг                                  |
| traffic_source_missing              | ✅ оставить  | Производный                                |
| traffic_source_grouped              | ✅ оставить  | Маркетинг                                  |
| utm_term_missing                    | ✅ оставить  | Производный                                |
| utm_term_grouped                    | ✅ оставить  | Маркетинг                                  |
| utm_sky_missing                     | ✅ оставить  | Производный                                |
| utm_sky_autotarget                  | ✅ оставить  | Маркетинг                                  |
| utm_sky_brand                       | ✅ оставить  | Маркетинг                                  |
| utm_sky_varicose                    | ✅ оставить  | Маркетинг                                  |
| utm_sky_sleep                       | ✅ оставить  | Маркетинг                                  |
| lead_group_missing                  | ⚠️ оставить | Производный                                |
| lead_group_grouped                  | ⚠️ оставить | Проверить                                  |
| problem_missing                     | ⚠️ оставить | Может зависеть от обработки                |
| problem_grouped                     | ⚠️ оставить | Текстовая агрегация                        |
| lead_height_known                   | ✅ оставить  | Производный                                |
| lead_height_log                     | ✅ оставить  | Производный                                |
| lead_height_bin                     | ✅ оставить  | Производный                                |
| delivery_cost_missing               | ✅ оставить  | Производный                                |
| delivery_cost_log                   | ✅ оставить  | Производный                                |
