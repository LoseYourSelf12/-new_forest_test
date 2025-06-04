from __future__ import annotations
from typing import Dict, Any
import os
from PIL import Image
from .memory import Memory

# Default OCR cache location, mirrors old project structure
_DEFAULT_CACHE_PATH = os.path.join('ocr', 'ocr_hash', 'ocr_hash.csv')
_memory = Memory(_DEFAULT_CACHE_PATH)

def calc_local_memory(file_path: str) -> Dict[str, Any] | None:
    """Load image and cached text if available."""
    if not os.path.isfile(file_path):
        return None
    try:
        img = Image.open(file_path)
    except Exception:
        return None
    txt = _memory.get(file_path)
    if txt is None:
        # placeholder for OCR reader; return empty text and cache it
        txt = ""
        _memory.add(file_path, txt)
    return {"img": img, "txt": txt, "file_name": os.path.basename(file_path)}


def split_categories_to_columns(df, ctgr_list):
    """Expand list of category flags into separate columns."""
    new_cols = {}
    for cat in ctgr_list:
        idx = cat.replace(" ", "")[:5]
        col = f"{idx}"
        new_cols[cat] = col
        df[col] = 0
    for i, row in df.iterrows():
        cell = row['category_present']
        if not isinstance(cell, list):
            cell = [cell]
        if len(cell) < len(ctgr_list):
            cell = cell + [0] * (len(ctgr_list) - len(cell))
        for j, cat in enumerate(ctgr_list):
            df.at[i, new_cols[cat]] = cell[j]
    return df
