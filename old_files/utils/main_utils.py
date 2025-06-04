import os
from typing import Dict, Any
import numpy as np

import base
from ocr import Reader
from .memorys import Memoryes
from .load_image import load_from_file


memory_ocr = Memoryes()


def calc_local_memory(file_path: str):
    """
    Кэширует распознанный текст баннера.
    """
    image = load_from_file(file_path)
    if image is None:
        return None
    hash_ocr = memory_ocr.load(file_path)
    file_name = os.path.basename(file_path)
    if hash_ocr is None:
        reader = Reader()
        content, diclaimer = reader.readbanner(image)
        content = ' '.join(content)
        txt = content + ' ' + diclaimer
        memory_ocr.append(file_path, txt)
        return {'img': image, 'txt': txt, 'file_name': file_name}
    else:
        return {'img': image, 'txt': hash_ocr, 'file_name': file_name}


def split_categories_to_columns(df, ctgr_list):
    from base.nameiddict import name_id_dict
    import pandas as pd
    new_cols = {}
    for cat in ctgr_list:
        cat_index = cat.replace(" ", "")[:5]
        cat_id = name_id_dict.get(cat, 0)
        col_name = f"{cat_index}_{cat_id}"
        new_cols[cat] = col_name
        df[col_name] = 0

    for i, row in df.iterrows():
        cell = row['category_present']
        if not isinstance(cell, list):
            cell = [cell]
        if len(cell) < len(ctgr_list):
            cell = cell + [0]*(len(ctgr_list)-len(cell))
        for j, cat in enumerate(ctgr_list):
            df.at[i, new_cols[cat]] = cell[j]
    return df