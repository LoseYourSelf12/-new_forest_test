import os
import pandas as pd
from collections import defaultdict

from base.nameiddict_cor import nameiddict_cor

from .datasets_utils import (
    get_file_size,
    dataset_generator,
    check_file_hash,
)


# Словарь с путями к датасетам
DATASET_PATHS = {
    "russ9200":  "/home/tbdbj/banners/russ9200",
    "russ2500":  "/home/tbdbj/banners/russ2500",
    "russ_apr_24":  "/home/tbdbj/banners/russ2024y/russ_apr_24",
    "russ_may_24":  "/home/tbdbj/banners/russ2024y/russ_may_24",
    "russ_jun_24":  "/home/tbdbj/banners/russ2024y/russ_jun_24",
    "russ_jul_24":  "/home/tbdbj/banners/russ2024y/russ_jul_24",
    "russ_aug_24":  "/home/tbdbj/banners/russ2024y/russ_aug_24",
    "russ_sep_24":  "/home/tbdbj/banners/russ2024y/russ_sep_24",
    "russ_oct_24":  "/home/tbdbj/banners/russ2024y/russ_oct_24",
    "russ_nov_24":  "/home/tbdbj/banners/russ2024y/russ_nov_24",
    "russ_dec_24":  "/home/tbdbj/banners/russ2024y/russ_dec_24",
    "russ_jan_24":  "/home/tbdbj/banners/russ2024y/russ_jan_24",
    "russ_feb_24":  "/home/tbdbj/banners/russ2024y/russ_feb_24",
    "russ_mar_24":  "/home/tbdbj/banners/russ2024y/russ_mar_24",
    "wb1000":  "/home/tbdbj/banners/marked_up_files/Пример_разметки_WB/1000_samples",
    'russ_check_1': '/home/tbdbj/banners/russ_check_1',
    'russ_1_half_year_2024': '/home/tbdbj/banners/russ_15_04/russ_1_half_year_2024',
    'russ_apr_2025': '/home/tbdbj/banners/russ_15_04/russ_apr_2025',
    'russ_feb_mar_2025': '/home/tbdbj/banners/russ_15_04/russ_feb_mar_2025',
    'russ_jul_oct_2024': '/home/tbdbj/banners/russ_15_04/russ_jul_oct_2024',
    'russ_nov_jan_2024_2025': '/home/tbdbj/banners/russ_15_04/russ_nov_jan_2024_2025'
}

# -----------------------------
# Генераторы датасетов
# -----------------------------

@dataset_generator("wb1000", DATASET_PATHS["wb1000"])
def ds_wb1000(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ9200", DATASET_PATHS["russ9200"])
def ds_russ9200(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ2500", DATASET_PATHS["russ2500"])
def ds_russ2500(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_apr_24", DATASET_PATHS["russ_apr_24"])
def ds_russ_apr_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_may_24", DATASET_PATHS["russ_may_24"])
def ds_russ_may_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_jun_24", DATASET_PATHS["russ_jun_24"])
def ds_russ_jun_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_jul_24", DATASET_PATHS["russ_jul_24"])
def ds_russ_jul_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_aug_24", DATASET_PATHS["russ_aug_24"])
def ds_russ_aug_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_sep_24", DATASET_PATHS["russ_sep_24"])
def ds_russ_sep_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_oct_24", DATASET_PATHS["russ_oct_24"])
def ds_russ_oct_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_nov_24", DATASET_PATHS["russ_nov_24"])
def ds_russ_nov_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_dec_24", DATASET_PATHS["russ_dec_24"])
def ds_russ_dec_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_jan_24", DATASET_PATHS["russ_jan_24"])
def ds_russ_jan_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_feb_24", DATASET_PATHS["russ_feb_24"])
def ds_russ_feb_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_mar_24", DATASET_PATHS["russ_mar_24"])
def ds_russ_mar_24(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_check_1", DATASET_PATHS["russ_check_1"])
def ds_russ_check_1(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_1_half_year_2024", DATASET_PATHS["russ_1_half_year_2024"])
def ds_russ_1_half_year_2024(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])
    
        
@dataset_generator("russ_apr_2025", DATASET_PATHS["russ_apr_2025"])
def ds_russ_apr_2025(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_feb_mar_2025", DATASET_PATHS["russ_feb_mar_2025"])
def ds_russ_feb_mar_2025(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_jul_oct_2024", DATASET_PATHS["russ_jul_oct_2024"])
def ds_russ_jul_oct_2024(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator("russ_nov_jan_2024_2025", DATASET_PATHS["russ_nov_jan_2024_2025"])
def ds_russ_nov_jan_2024_2025(csv_path=None):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])

# -----------------------------
# Функции объединения
# -----------------------------

def ds_combine(*dataset_generators):
    """
    Объединяет несколько датасетов с учётом дубликатов (используя размер файла),
    и убирает повторяющиеся категории. 
    Дубликатом считаем файл с одинаковым размером и file_name из разметки.
    """
    storage = {}

    for gen_func in dataset_generators:
        for file_name, category, current_file_name in gen_func():
            cat_list = []
            # Разделяем категории и делаем множеством для удаления дубликатов
            if category.strip():
                cat_list = [c.strip() for c in category.split(',')]
            cat_set = set(cat_list)
            size = get_file_size(current_file_name)
            key = (file_name, size)
            # Создаем пустую запись для файла
            if key not in storage:
                storage[key] = {
                    'categories': set(),
                    'rows': []
                }
            # обновляем запись
            storage[key]['categories'].update(cat_set)
            storage[key]['rows'].append((file_name, category, current_file_name))  # Может быть несколько записей?

    results = []
    for (file_name, size), data_dict in storage.items():
        rows = data_dict['rows']
        combined_cat_set = data_dict['categories']

        # Считаем количество дубликатов и сколько удаляем их
        category_counter = {}
        for (_, cat_str, _) in rows:
            category_counter.setdefault(cat_str, 0)
            category_counter[cat_str] += 1
        duplicates_removed = 0
        for cat_str, count in category_counter.items():
            if count > 1:
                duplicates_removed += (count - 1)

        if duplicates_removed > 0:
            print(f"Удалено {duplicates_removed} дубликатов для file_name={file_name}, size={size}")
            
        combined_cat_list = sorted(list(combined_cat_set)) # сортируем для единообразия
        combined_categories_str = ",".join(combined_cat_list) # Объединяем обратно в строку

        # Возьмём current_file_name из первой записи, учитывая, что дубликатов уже нет
        current_file_name = rows[0][2] if rows else None
        results.append((file_name, combined_categories_str, current_file_name))

    for item in results:
        yield item


def ds_russ2024y():
    """
    Генератор, объединяющий все 12 'russ_..._24' датасетов.
    """
    combined = ds_combine(
        ds_russ_apr_24,
        ds_russ_may_24,
        ds_russ_jun_24,
        ds_russ_jul_24,
        ds_russ_aug_24,
        ds_russ_sep_24,
        ds_russ_oct_24,
        ds_russ_nov_24,
        ds_russ_dec_24,
        ds_russ_jan_24,
        ds_russ_feb_24,
        ds_russ_mar_24
    )
    for row in combined:
        yield row


def ds_russ_15_04():
    combined = ds_combine(
        ds_russ_1_half_year_2024,
        ds_russ_apr_2025,
        ds_russ_feb_mar_2025,
        ds_russ_jul_oct_2024,
        ds_russ_nov_jan_2024_2025,
    )
    for row in combined:
        yield row


def ds_check_ctgr(ctgr_names: list, ds):
    # Получаем список синонимов для каждой категории
    ctgr_synonyms_list = [nameiddict_cor.get(cat, [cat]) for cat in ctgr_names]

    for file_name, categories_str, current_file_name in ds():
        if len(ctgr_names) == 1:
            flag = 1 if any(s in categories_str for s in ctgr_synonyms_list[0]) else 0
            yield file_name, flag, current_file_name
        else:
            flags = []
            for syn_list in ctgr_synonyms_list:
                flag = 1 if any(s in categories_str for s in syn_list) else 0
                flags.append(flag)
            yield file_name, flags, current_file_name


def ds_num_iter(num, yield_ds):
    """
    Возвращает первые 'num' записей из генератора yield_ds.
    """
    for i, data in enumerate(yield_ds):
        if i >= num:
            break
        yield data
