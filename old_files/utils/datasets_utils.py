import os
import re
from collections import defaultdict
from base.nameiddict_cor import nameiddict_cor


diffs = []

def is_image_file(file_name: str) -> bool:
    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    return any(file_name.lower().endswith(ext) for ext in allowed_extensions)


def get_file_size(current_file_name: str):
    if not current_file_name or not os.path.isfile(current_file_name):
        return None
    return os.path.getsize(current_file_name)


def full_image_path(file_name: str, image_folders: list = None):
    if image_folders is None:
        image_folders = ["/home/tbdbj/banners/marked_up_files"]
    
    for folder in image_folders:
        for dirpath, _, filenames in os.walk(folder):
            if file_name in filenames:
                return os.path.abspath(os.path.join(dirpath, file_name))
    return None


def auto_csv_path(dataset_name: str, base_path: str) -> str:
    return os.path.join(base_path, "markup", f"labels_{dataset_name}.csv")


def filtered_generator(generator_func):
    def wrapper(*args, **kwargs):
        for file_name, category, current_file_name in generator_func(*args, **kwargs):
            if is_image_file(file_name):
                yield file_name, category, current_file_name
            else:
                print(f"Пропущен файл не-изображение: {file_name}")
                pass
    return wrapper


def dataset_generator(dataset_name: str, dataset_path: str):
    def decorator(generator_func):
        generator_func._dataset_name = dataset_name
        def wrapper(*args, **kwargs):
            # Если csv_path не указан, сформируем автоматически
            if 'csv_path' not in kwargs or not kwargs['csv_path']:
                kwargs['csv_path'] = auto_csv_path(dataset_name, dataset_path)

            for file_name, category, current_file_name in generator_func(*args, **kwargs):
                full_path = full_image_path(current_file_name, [dataset_path])
                if is_image_file(file_name):
                    yield file_name, category, full_path
                else:
                    print(f"Пропущен не-изображение: {file_name}")
        return wrapper
    return decorator


def use_diff_file(file_path: str):
    global diffs
    current_diff = {"category": None, "modifications": {}}
    with open(file_path, 'r', encoding='utf-8') as f:
        header = True
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header:
                header = False
                parts = line.split(maxsplit=1)
                current_diff["category"] = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Ожидается формат: "имя_файла <+ или ->"
                fname = line[:-2].strip()
                op = line[-1]
                if op == '+':
                    current_diff["modifications"][fname] = True
                elif op == '-':
                    current_diff["modifications"][fname] = False
                else:
                    raise Exception(f"Неверный формат строки в файле {file_path}: {line}")
    diffs.append(current_diff)


def load_diff_files(file_list=None, directory="/home/tbdbj/forest_test/diff_files"):
    """
    Загружает diff-файлы. Если file_list не задан, ищет файлы по шаблону 'labels_*.txt'
    в указанной директории.
    """
    import glob
    if file_list is None:
        file_list = glob.glob(os.path.join(directory, "labels_*.txt"))
    for file_path in file_list:
        use_diff_file(file_path)


def check_file_hash(ds_name, file_name, ctgr_str, current_file_name):
    ctgr_list = ctgr_str.split(",") if ctgr_str else []
    for diff in diffs:
        if file_name in diff["modifications"]:
            synonyms = nameiddict_cor.get(diff["category"], [diff["category"]])
            if diff["modifications"][file_name]:  # True: добавление категории
                if not any(s in ctgr_list for s in synonyms):
                    ctgr_list.append(diff["category"])
                    print(f"Добавлена категория {diff['category']} для файла {file_name}")
            else:  # False: удаление категории
                removed = False
                for s in synonyms:
                    if s in ctgr_list:
                        ctgr_list.remove(s)
                        removed = True
                if removed:
                    print(f"Удалены категории {synonyms} для файла {file_name}")
                else:
                    print(f"Категория {synonyms} для файла {file_name} не найдена")
    new_ctgr_str = ",".join(ctgr_list)
    return file_name, new_ctgr_str, current_file_name
