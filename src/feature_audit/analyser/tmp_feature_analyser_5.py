def width_category(x):
    if pd.isna(x):
        return "unknown"
    elif x < 10 or x > 50:
        return "anomaly"
    elif x < 25:
        return "small"
    elif x <= 35:
        return "normal"
    else:
        return "large"

df["width_cat"] = df["lead_Ширина"].apply(width_category)

df = df.drop(columns=["lead_Ширина"])

####

df["lead_height_known"] = df["lead_Линейная высота (см)"].notna().astype(int)
df = df.drop(columns=["lead_Линейная высота (см)"])

####

def payment_type(x):
    if pd.isna(x):
        return "unknown"
    elif x == "Наложенный платеж":
        return "cash_on_delivery"
    else:
        return "online"

df["lead_payment_type"] = df["lead_Вид оплаты"].apply(payment_type)
df = df.drop(columns=["lead_Вид оплаты"])

############

# returned_dt -утечка, просто удаляем
df = df.drop(columns=["returned_dt"])

############

def delivery_type(x):
    if pd.isna(x):
        return "unknown"
    elif x == "СДЭК до ПВЗ":
        return "pickup_point"
    elif x in ["СДЭК до Двери", "Курьер ЕМС"]:
        return "door_delivery"
    elif x == "Почта":
        return "post"
    elif x == "Самовывоз":
        return "pickup_point"  # логично объединить
    else:
        return "unknown"  # на случай мусора

df["lead_delivery_type"] = df["lead_Служба доставки"].apply(delivery_type)
df = df.drop(columns=["lead_Служба доставки"])

###########

# разница мала, и большой дисбаланс значений
df = df.drop(columns=["lead_Компания Отправитель"])

###########

#низкая кардинальность, всего ~5 значений, достаточное количество наблюдений
# есть порядок 0.80 → 0.83 → 0.85 → 0.88

df["lead_group_id"] = df["lead_group_id"].astype("object").fillna("unknown").astype("category")

group_means = df.groupby("lead_group_id")["buyout_flag"].mean()
df["lead_group_quality"] = df["lead_group_id"].map(group_means)
df = df.drop(columns=["lead_group_id"])

##############

# слишком много пропусков
df["lead_mass_known"] = df["lead_Масса (гр)"].notna().astype(int)

# Ассиметрия, большие выбросы и разбросы
# есть ли масса бинарный сигнал, насколько большая  количественный сигнал
df["lead_mass_log"] = np.log1p(df["lead_Масса (гр)"])

df["lead_mass_log"] = df["lead_mass_log"].fillna(0)
df = df.drop(columns=["lead_Масса (гр)"])

#######
# утечка
df = df.drop(columns=["lead_closed_dt"])

#######
df["contact_created_dt"] = pd.to_datetime(df["contact_created_dt"], errors="coerce")
df["contact_updated_dt"] = pd.to_datetime(df["contact_updated_dt"], errors="coerce")

# Потенциально отражает, скорость обработки лида, активность менеджера, интерес клиента
df["contact_update_delay_days"] = (
    df["contact_updated_dt"] - df["contact_created_dt"]
).dt.total_seconds() / (3600 * 24)

# 5 пропусков
df["contact_update_delay_days"] = df["contact_update_delay_days"].fillna(
    df["contact_update_delay_days"].median()
)
df["contact_update_delay_log"] = np.log1p(df["contact_update_delay_days"])
df = df.drop(columns=["contact_created_dt", "contact_updated_dt", "contact_update_delay_days"])

##############

# Смотрим платный трафик или бесплатный CPC = Cost Per Click и CPM = Cost Per Mille (1000 показов). Это индикатор платного привлечения клиента, тут еще можно поэкспериментировать
# если время будет
df["is_paid_traffic"] = df["lead_utm_medium"].astype(str).str.lower().str.contains("cpc|cpm").astype(int)
df = df.drop(columns=["lead_utm_medium"])

###########

df["lead_Категория и варианты выбора"] = df["lead_Категория и варианты выбора"].fillna("unknown").replace({"Нет категории": "unknown"})
freq = df["lead_Категория и варианты выбора"].value_counts(normalize=True)
df["lead_category_freq"] = df["lead_Категория и варианты выбора"].map(freq)
df = df.drop(columns=["lead_Категория и варианты выбора"])

##########
# received_dt -утечка, просто удаляем, факт завершённой сделки
df = df.drop(columns=["received_dt"])

############

df["lead_Модель телефона"] = df["lead_Модель телефона"].fillna("unknown")
df["is_feature_phone"] = (df["lead_Модель телефона"] == "Кнопочный").astype(int)
# todo сделать target encoding
df = df.drop(columns=["lead_Модель телефона"])

########

#Итого признак "lead_Дата перехода Передан в доставку" разреженный
# дубль handed_to_delivery_dt, удаляем
df = df.drop(columns=["lead_Дата перехода Передан в доставку"])

#########

# Производные признаки
#Время до передачи в доставку
df["lead_created_dt"] = pd.to_datetime(df["lead_created_dt"], errors='coerce')
df["lead_to_delivery_days"] = (
    df["handed_to_delivery_dt"] - df["lead_created_dt"]
).dt.total_seconds() / 86400

#Время от продажи до доставки
df["sale_date_dt"] = pd.to_datetime(df["sale_date_dt"], errors='coerce')
df["sale_to_delivery_days"] = (
    df["handed_to_delivery_dt"] - df["sale_date_dt"]
).dt.total_seconds() / 86400

df["sale_to_delivery_log"] = np.log1p(df["sale_to_delivery_days"])
df["sale_to_delivery_log"].describe()

# Корректное заполнение пропусков, -1 нет события
df["sale_to_delivery_missing"] = df["sale_to_delivery_log"].isna().astype(int)
df["sale_to_delivery_log"] = df["sale_to_delivery_log"].fillna(-1)

# Корректное заполнение пропусков, -1 нет события
df["lead_to_delivery_days_missing"] = df["lead_to_delivery_days_log"].isna().astype(int)
df["lead_to_delivery_days_log"] = df["lead_to_delivery_days_log"].fillna(-1)
df["lead_to_delivery_days_log"].describe()

#!!!!! sale_date_dt, lead_created_dt - написать assert на наличие данных
# полей

df = df.drop(columns=["lead_to_delivery_days", "sale_to_delivery_days",
                      "handed_to_delivery_dt"])

##########

# Уберем мусор
df["lead_utm_campaign"] = (df["lead_utm_campaign"].replace
                           (["{campaing_id}", "Неизвестно"], np.nan))

# Признак сильно разреженный, отмечаем где не заполнен
df["lead_utm_campaign_missing"] = df["lead_utm_campaign"].isna().astype(int)

# В теории можно Target Encoding, но пока простейший вариант, хвост
# сократить
top_k = 20  # можно 20–50
top_values = df["lead_utm_campaign"].value_counts().head(top_k).index

df["lead_utm_campaign_grouped"] = df["lead_utm_campaign"].where(
    df["lead_utm_campaign"].isin(top_values),
    "unknown"
)

df = df.drop(columns=["lead_utm_campaign"])

##############

df["traffic_source_missing"] = df["contact_Источник трафика"].isna().astype(int)

top_k = 20

top_values = (
    df["contact_Источник трафика"]
    .value_counts()
    .head(top_k)
    .index
)

df["traffic_source_grouped"] = df["contact_Источник трафика"].where(
    df["contact_Источник трафика"].isin(top_values),
    "other"
)


df = df.drop(columns=["contact_Источник трафика"])

##############

# --- 1. приведение к datetime + убираем tz ---
df["lead_created_dt"] = pd.to_datetime(df["lead_created_dt"], errors="coerce").dt.tz_localize(None)

df["lead_Дата перехода в Сборку"] = pd.to_datetime(
    df["lead_Дата перехода в Сборку"], errors="coerce"
).dt.tz_localize(None)

start = pd.Timestamp("2025-03-01")
end   = pd.Timestamp("2026-03-29")

date_cols = [
    "lead_created_dt",
    "lead_Дата перехода в Сборку"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

    df.loc[
        (df[col] < start) | (df[col] > end),
        col
    ] = pd.NaT

# --- 2. delay (сборка - создание) ---
df["lead_to_assembly_days"] = (
    df["lead_Дата перехода в Сборку"] - df["lead_created_dt"]
).dt.total_seconds() / 86400

# --- 3. исправление timezone-сдвига (+1 день) ---
df["lead_to_assembly_days"] = df["lead_to_assembly_days"] + 1

# --- 4. флаг события ---
df["assembly_flag"] = df["lead_Дата перехода в Сборку"].notna().astype(int)

# --- 5. контроль хвоста ---
df["lead_to_assembly_long"] = (df["lead_to_assembly_days"] > 30).astype(int)

df["lead_to_assembly_days"] = df["lead_to_assembly_days"].clip(upper=30)

# --- 6. обработка пропусков ---
df["lead_to_assembly_days"] = df["lead_to_assembly_days"].fillna(-1)

# --- 7. удаление исходного признака ---
df = df.drop(columns=["lead_Дата перехода в Сборку"])

###########

df["lead_utm_term"] = df["lead_utm_term"].replace(
    ["Неизвестно"],
    np.nan
)

df["utm_term_missing"] = df["lead_utm_term"].isna().astype(int)

# --- 3. top-K + other ---
top_k = 40
top_values = df["lead_utm_term"].value_counts().head(top_k).index

df["utm_term_grouped"] = df["lead_utm_term"].where(
    df["lead_utm_term"].isin(top_values),
    "other"
)

df = df.drop(columns=["lead_utm_term"])

##############

df["lead_utm_sky"] = df["lead_utm_sky"].replace(
    ["{keyword}"],
    np.nan
)

# Базовый missing-флаг
df["utm_sky_missing"] = df["lead_utm_sky"].isna().astype(int)

# отдельный тип трафика  часто сильно влияет
df["utm_sky_autotarget"] = (df["lead_utm_sky"] == "---autotargeting").astype(int)

# выделим бренд
df["utm_sky_brand"] = df["lead_utm_sky"].str.contains(
    "artraid|артрейд",
    case=False,
    na=False
).astype(int)

# семантика запроса
df["utm_sky_varicose"] = df["lead_utm_sky"].str.contains(
    "варикоз|вены|флеболог",
    case=False,
    na=False
).astype(int)

df["utm_sky_sleep"] = df["lead_utm_sky"].str.contains(
    "сон|sleep",
    case=False,
    na=False
).astype(int)

df = df.drop(columns=["lead_utm_sky"])

############

# rejected_dt - утечка, просто удаляем
df = df.drop(columns=["rejected_dt"])

##############

# флаг пропуска
df["lead_group_missing"] = df["lead_group"].isna().astype(int)

# топ-2 группы + other
main_groups = ["yur", "but"]

df["lead_group_grouped"] = df["lead_group"].where(
    df["lead_group"].isin(main_groups),
    "other"
)

# удаление исходного столбца
df = df.drop(columns=["lead_group"])

#############

# missing-флаг
df["problem_missing"] = df["lead_Проблема"].isna().astype(int)

# --- 3. укрупнение категорий ---
main_groups = [
    "Суставы и позвоночник",
    "Варикоз",
    "Сердечно-сосудистые заболевания",
    "Бессоница",
    "Головные боли",
    "Отеки",
    "Зрительная система",
    "Давление",
    "Инсульт",
    "Боли и тяжесть в ногах"
]

df["problem_grouped"] = df["lead_Проблема"].where(
    df["lead_Проблема"].isin(main_groups),
    "other"
)

df = df.drop(columns=["lead_Проблема"])