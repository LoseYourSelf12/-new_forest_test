import pandas as pd
import glob

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import re
import ast
import os
from base.nameiddict_cor import nameiddict_cor
from collections import defaultdict

dataset_name = ""
files_hash = {}
ctgr_name = ""


def is_image_file(file_name):
    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    return any(file_name.lower().endswith(ext) for ext in allowed_extensions)


def auto_csv_path(dataset_name: str) -> str:
    """
    Автоматически формирует путь к файлу разметки для указанного датасета,
    исходя из соглашения: <DATASET_PATH>/markup/labels_<dataset_name>.csv
    """
    base_path = DATASET_PATHS[dataset_name]
    return os.path.join(base_path, "markup", f"labels_{dataset_name}.csv")


# Старая версия
def filtered_generator(generator_func):
    def wrapper(*args, **kwargs):
        for file_name, category, current_file_name in generator_func(*args, **kwargs):
            if is_image_file(file_name):
                yield file_name, category, current_file_name
            else:
                print(f"Пропущен файл не-изображение: {file_name}")
                pass
    return wrapper


def dataset_generator(dataset_name: str):
    def decorator(generator_func):
        def wrapper(*args, **kwargs):
            # Если csv_path не указан, сформируем автоматически
            if 'csv_path' not in kwargs or not kwargs['csv_path']:
                kwargs['csv_path'] = auto_csv_path(dataset_name)

            # Вызываем исходный генератор
            for file_name, category, current_file_name in generator_func(*args, **kwargs):
                full_path = full_image_path(file_name, [DATASET_PATHS[dataset_name]])

                # Проверим, не является ли файл не-изображением
                if is_image_file(file_name):
                    yield file_name, category, full_path
                else:
                    print(f"Пропущен не-изображение: {file_name}")
                    continue
        return wrapper
    return decorator


def use_diff_file(file_name:str):
    global dataset_name, files_hash, ctgr_name
    
    with open(file_name, 'r') as file:
        header = True
        for line in file:
            line = line.strip()
            if line == "" or line[0] == '#':
                continue
            
            if header:
                header = False
                dataset_name = line.split(' ')[0]
                ctgr_name = line[len(dataset_name):].strip()
            else:
                fname = line[:-2].strip()
                if line[-1] == '+':
                    files_hash[fname] = True
                elif line [-1] == '-':
                    files_hash[fname] = False
                else:
                    raise Exception(f"Неверный формат файла:\nfile= {file_name}\nline={line}")


def check_file_hash(ds_name, file_name, ctgr_str, current_file_name):
    global dataset_name, files_hash, ctgr_name
    
    ctgr_list = ctgr_str.split(",") if ctgr_str else []
    if dataset_name == ds_name and file_name in files_hash:
        if files_hash[file_name]: # true + false -   
            if ctgr_name in ctgr_list:
                print(f"Категория {ctgr_name} в файле {file_name} уже присвоена")
            else:
                ctgr_list.append(ctgr_name)                    
        else:
            if ctgr_name in ctgr_list:
                ctgr_list.remove[ctgr_name]
            else:
                print(f"Категория {ctgr_name} в файле {file_name} не присвоена")
    
    ctgr_str = ",".join(ctgr_list)
    
    return file_name, ctgr_str, current_file_name


def get_file_size(current_file_name: str) -> int:
    if not current_file_name or not os.path.isfile(current_file_name):
        return None
    return os.path.getsize(current_file_name)


def ds_combine(*dataset_generators):
    storage = {}

    # Считываем все строки из генераторов
    for gen_func in dataset_generators:
        for file_name, category, current_file_name in gen_func():
            size = get_file_size(current_file_name)
            key = (file_name, size)
            if key not in storage:
                storage[key] = {
                    'categories': set(),
                    'rows': []
                }
            storage[key]['rows'].append((file_name, category, current_file_name))
            # Категорию добавим позже (когда будем анализировать дубликаты)

    # Готовим структуру для итоговых строк
    results = []

    for (file_name, size), data_dict in storage.items():
        rows = data_dict['rows']

        # Собираем все категории, которые встретились у этих (file_name, size)
        # Параллельно считаем, сколько было точных дублей.
        category_counter = {}
        # (category -> [ (file_name, category, current_file_name), ... ])
        
        for (f, cat, cfn) in rows:
            category_counter.setdefault(cat, []).append((f, cat, cfn))
        
        # Отчёт о дубликатах
        duplicates_removed = 0
        for cat, cat_rows in category_counter.items():
            if len(cat_rows) > 1:
                # было несколько дублей полностью одинаковых (f, cat, size)
                duplicates_removed += (len(cat_rows) - 1)

        # Если категория у нас одна или несколько разных, их надо объединить в одну строку.
        unique_cats = list(category_counter.keys())
        if len(unique_cats) > 1:
            # Частичное совпадение: file_name и size те же, но категории разные
            # Нужно объединить категории через запятую
            print(f"Объединяем категории для file_name={file_name}, size={size}: {unique_cats}")
        if duplicates_removed > 0:
            print(f"Удалено {duplicates_removed} дубликатов для file_name={file_name}, size={size}")

        combined_categories = ",".join(sorted(unique_cats))

        final_current_file_name = rows[0][2] if rows else None

        # Собираем итоговую запись
        results.append((file_name, combined_categories, final_current_file_name))

    
    name_to_sizes = defaultdict(list)
    for (f, s), _data in storage.items():
        name_to_sizes[f].append(s)
    for f, sizes in name_to_sizes.items():
        # убираем None, если встречается
        non_null_sizes = [sz for sz in sizes if sz is not None]
        # Если у этого file_name более одного уникального размера, делаем пометку.
        if len(set(non_null_sizes)) > 1:
            print(f"[INFO] У файла file_name={f} есть несколько вариантов размеров: {set(non_null_sizes)}. Оставляем все.")

    for item in results:
        yield item


@dataset_generator 
def ds_wb1000(csv_path='/home/tbdbj/banners/marked_up_files/Пример_разметки_WB/1000_samples/labels.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():  
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])
   

@dataset_generator     
def ds_russ9200(csv_path='/home/tbdbj/banners/russ9200_corrected/markup/labels_russ9200.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator     
def ds_russ2500(csv_path='/home/tbdbj/banners/russ2500_corrected/markup/labels_russ2500.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_apr_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_apr_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_aug_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_aug_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_dec_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_dec_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_feb_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_feb_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_jan_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_jan_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_jul_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_jul_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_jun_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_jun_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_mar_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_mar_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_may_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_may_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_nov_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_nov_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_oct_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_oct_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])


@dataset_generator 
def ds_russ_sep_24(csv_path='/home/tbdbj/banners/russ2024y_corrected/markup/labels_russ_sep_24.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        yield check_file_hash(row['dataset'], row['file_name'], row['category'], row['current_file_name'])

def ds_russ2024y():
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
        
def ds_check_ctgr(ctgr_name, ds):
    ctgr_list = nameiddict_cor.get(ctgr_name, [ctgr_name])
    for file_name, categoryes, current_file_name in ds():
        if any(ctgr in categoryes for ctgr in ctgr_list):
            yield file_name, 1, current_file_name
        else:
            yield file_name, 0, current_file_name
          
            
def ds_num_iter(num, yield_ds):
    for i, data in enumerate(yield_ds):
        if i > num:
            break
        yield data


def load_df_from_csv(*csv):
    file_paths = []
    for path in csv:
        file_paths.extend(glob.glob(path))
    df_list = [pd.read_csv(file) for file in file_paths]
    if len(file_paths) > 1:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = df_list[0]
    df.dropna(inplace=True)
    return df


def load_X_Y_from_csv(test_ctgr, df):
    from ocr import OcrDetector
    from yolo import YOLODetector
    df_loaded = df
    df_loaded['detection_vector'] = df_loaded['detection_vector'].apply(ast.literal_eval)

    feature_names = []
    for detector in test_ctgr.list:
        if isinstance(detector, YOLODetector):  # YOLO
            feature_names.extend([f"YOLO_{name}" for name in detector.names])
        elif isinstance(detector, OcrDetector):  # OCR
            feature_names.extend([f"OCR_{name.upper()}" for name in detector.names])

    detection_df = pd.DataFrame(df_loaded['detection_vector'].tolist(), columns=feature_names)

    X = detection_df
    Y = df_loaded['category_present']

    return X, Y


def print_metrix(df=None, Y_t=None, Y_p=None):
    if Y_t is not None and Y_p is not None:
        Y_test = Y_t
        Y_pred = Y_p
    else:
        Y_test = df['category_present'].astype(int)
        Y_pred = df['threshold_passed']
    try:
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='binary')
        recall = recall_score(Y_test, Y_pred, average='binary')
        f1 = f1_score(Y_test, Y_pred, average='binary')
        roc_auc = roc_auc_score(Y_test, Y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
    except ValueError:
        roc_auc = roc_auc_score(Y_test, Y_pred)
        print(f"ROC AUC Score: {roc_auc:.4f}")
     
        
def feature_importances_calc(mixer, X_train):
    feature_importances = mixer.model.feature_importances_
    feature_names = X_train.columns
    
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[sorted_idx]
    sorted_names = feature_names[sorted_idx]

    plt.figure(figsize=(12, 6))
    plt.barh(sorted_names, sorted_importances, color="skyblue")
    plt.xlabel(f"Важность признаков {mixer.name}")
    plt.title(f"График важности признаков {mixer.name}")
    plt.gca().invert_yaxis()

    plt.savefig(f"results_feature_importances/{mixer.name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return feature_importances, feature_names


def safe_filename(name):
    return re.sub(r'[\/:*?"<>|]', '_', name)


def full_image_path(file_name: str, image_folders: list = None):
    """
    Ищет файл file_name в указанных директориях.
    Если image_folders не задан, используется значение по умолчанию.
    """
    # Если не передали список директорий, используем глобальную папку
    if image_folders is None:
        image_folders = ["/home/tbdbj/banners/marked_up_files"]
    
    for folder in image_folders:
        for dirpath, _, filenames in os.walk(folder):
            if file_name in filenames:
                return os.path.abspath(os.path.join(dirpath, file_name))
    return None
