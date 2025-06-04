import os 
import shutil 
import datetime 
import pandas as pd 
from PIL import Image

from utils import ds_combine, load_diff_files 

def export_dataset(dataset_generators, new_dataset_name, dataset_names): 
    if len(dataset_generators) != len(dataset_names): 
        raise ValueError("Количество генераторов должно совпадать с количеством названий датасетов")

    combined_results = []
    for gen_func, ds_name in zip(dataset_generators, dataset_names):
        results = list(ds_combine(gen_func))
        for file_name, category, current_file_name in results:
            combined_results.append((file_name, category, current_file_name, ds_name))

    total_count = len(combined_results)
    today_str = datetime.datetime.now().strftime("%d_%m_%Y")
    out_dir = f"/home/tbdbj/banners/datasets_output/{new_dataset_name}_{today_str}"
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    records = []
    missing_cat_records = []
    missing_files = 0
    broken_files = 0

    for file_name, categories, current_file_name, ds_name in combined_results:
        if not current_file_name or not os.path.isfile(current_file_name):
            missing_files += 1
            continue
        try:
            with Image.open(current_file_name) as img:
                img.verify()
        except Exception as e:
            broken_files += 1
            continue
        
        base_name = os.path.basename(current_file_name)
        dst_path = os.path.join(data_dir, base_name)
        shutil.copy2(current_file_name, dst_path)
        
        record = {
            'current_file_name': base_name,
            'file_name': file_name,
            'category': categories,
            'dataset': ds_name
        }
        if not categories.strip():
            missing_cat_records.append(record)
        else:
            records.append(record)

    csv_name = f"labels_{new_dataset_name}.csv"
    csv_path = os.path.join(out_dir, csv_name)
    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_path, index=False, encoding='utf-8', sep=';')

    if missing_cat_records:
        csv_name_no_cat = f"labels_{new_dataset_name}_no_category.csv"
        csv_path_no_cat = os.path.join(out_dir, csv_name_no_cat)
        df_no_cat = pd.DataFrame(missing_cat_records)
        df_no_cat.to_csv(csv_path_no_cat, index=False, encoding='utf-8', sep=';')

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as rep:
        rep.write(f"New dataset: {new_dataset_name}\n")
        rep.write(f"Date: {today_str}\n")
        rep.write(f"Total items from datasets: {total_count}\n")
        rep.write(f"Copied files: {len(records)}\n")
        rep.write(f"Missing (file not found): {missing_files}\n")
        rep.write(f"Broken files: {broken_files}\n")

    print(f"Выгрузка завершена для '{new_dataset_name}'. Результат в: {out_dir}")

