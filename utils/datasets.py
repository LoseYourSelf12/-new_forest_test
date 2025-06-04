from __future__ import annotations
import os
import glob
import re
from typing import Iterator, Iterable, Tuple, List
import pandas as pd
from .nameiddict_cor import nameiddict_cor

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
    "russ_check_1": "/home/tbdbj/banners/russ_check_1",
    "russ_1_half_year_2024": "/home/tbdbj/banners/russ_15_04/russ_1_half_year_2024",
    "russ_apr_2025": "/home/tbdbj/banners/russ_15_04/russ_apr_2025",
    "russ_feb_mar_2025": "/home/tbdbj/banners/russ_15_04/russ_feb_mar_2025",
    "russ_jul_oct_2024": "/home/tbdbj/banners/russ_15_04/russ_jul_oct_2024",
    "russ_nov_jan_2024_2025": "/home/tbdbj/banners/russ_15_04/russ_nov_jan_2024_2025",
}

diffs: List[dict] = []

class Dataset:
    def __init__(self, name: str, base_path: str) -> None:
        self.name = name
        self.base_path = base_path

    def csv_path(self) -> str:
        return os.path.join(self.base_path, "markup", f"labels_{self.name}.csv")

    def _find_image(self, fname: str) -> str:
        for root, _, files in os.walk(self.base_path):
            if fname in files:
                return os.path.join(root, fname)
        return os.path.join(self.base_path, fname)

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        path = self.csv_path()
        df = pd.read_csv(path, sep=None, engine="python")
        file_col = "file_name"
        cur_col = "current_file_name" if "current_file_name" in df.columns else file_col
        for _, row in df.iterrows():
            file_name = str(row[file_col])
            category = str(row.get("category", ""))
            current_file = str(row.get(cur_col, row[file_col]))
            file_name, category, current_file = apply_diffs(file_name, category, current_file)
            yield file_name, category, self._find_image(current_file)

def split_categories(ctgr_str: str) -> List[str]:
    if not ctgr_str:
        return []
    matches = list(re.finditer(r"\d{2}\.\d{2}", ctgr_str))
    if not matches:
        return [ctgr_str.strip()]
    result = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ctgr_str)
        result.append(ctgr_str[start:end].strip(" ,;"))
    return result


def apply_diffs(file_name: str, ctgr_str: str, current_file: str) -> Tuple[str, str, str]:
    cat_list = split_categories(ctgr_str)
    for diff in diffs:
        if file_name in diff["modifications"]:
            synonyms = nameiddict_cor.get(diff["category"], [diff["category"]])
            if diff["modifications"][file_name]:
                if not any(s in cat_list for s in synonyms):
                    cat_list.append(diff["category"])
            else:
                cat_list = [c for c in cat_list if c not in synonyms]
    return file_name, ",".join(cat_list), current_file


def load_diff_files(file_list: Iterable[str] | None = None, directory: str = "diff_files") -> None:
    global diffs
    if file_list is None:
        file_list = glob.glob(os.path.join(directory, "labels_*.txt"))
    for path in file_list:
        use_diff_file(path)


def use_diff_file(file_path: str) -> None:
    current = {"category": None, "modifications": {}}
    with open(file_path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header:
                header = False
                parts = line.split(maxsplit=1)
                current["category"] = parts[1].strip() if len(parts) > 1 else ""
            else:
                fname = line[:-2].strip()
                op = line[-1]
                current["modifications"][fname] = True if op == "+" else False
    diffs.append(current)


def combine(*datasets: Dataset) -> Iterator[Tuple[str, str, str]]:
    storage: dict[Tuple[str, int | None], dict] = {}
    for ds in datasets:
        for file_name, category, current_file in ds:
            cat_set = set(split_categories(category))
            size = os.path.getsize(current_file) if os.path.isfile(current_file) else None
            key = (file_name, size)
            if key not in storage:
                storage[key] = {"categories": set(), "rows": []}
            storage[key]["categories"].update(cat_set)
            storage[key]["rows"].append((file_name, category, current_file))
    for (file_name, size), data in storage.items():
        combined = ",".join(sorted(data["categories"]))
        current_file = data["rows"][0][2]
        yield file_name, combined, current_file

def ds_check_ctgr(ctgr_names: List[str], ds):
    syn_list = [nameiddict_cor.get(cat, [cat]) for cat in ctgr_names]
    for file_name, categories_str, current_file in ds():
        if len(ctgr_names) == 1:
            flag = 1 if any(s in categories_str for s in syn_list[0]) else 0
            yield file_name, flag, current_file
        else:
            flags = []
            for synonyms in syn_list:
                flags.append(1 if any(s in categories_str for s in synonyms) else 0)
            yield file_name, flags, current_file


def ds_num_iter(num: int, yield_ds):
    for i, data in enumerate(yield_ds):
        if i >= num:
            break
        yield data
