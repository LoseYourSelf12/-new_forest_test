from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from .base import Detector
from .retextfinder import ReTextFinder


def load_qwen_text(file_id: str, csv_path: str, delimiter: str = ',') -> str:
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', delimiter=delimiter)
        row = df.loc[df['file_name'] == file_id]
        if not row.empty:
            return str(row.iloc[0]['texts'])
    except Exception as e:
        print(f"Ошибка при чтении CSV файла qwen: {e}")
    return ''


class QwenDetector(Detector):
    """Detector working with preprocessed Qwen text."""

    def __init__(self, name: str, keywords: List[List[str]] | List[str], qwen_file: str, delimiter: str = ','):
        if all(isinstance(item, list) for item in keywords):
            flat = [word for sub in keywords for word in sub]
            original = keywords
        else:
            flat = list(keywords)
            original = [keywords]
        super().__init__(name=name, features=flat)
        self._finder = ReTextFinder(original)
        self._qwen_file = qwen_file
        self._delimiter = delimiter

    def __call__(self, local: Dict[str, Any]) -> None:
        file_id = local.get('current_file_name', local.get('file_name'))
        text = local.get('qwen_text')
        if text is None:
            if file_id is None:
                self._vec = np.zeros(len(self._features), dtype=float)
                return
            text = load_qwen_text(file_id, self._qwen_file, self._delimiter)
        vec = np.zeros(len(self._features), dtype=float)
        for idx in self._finder.finditer(text):
            vec[idx] = 1.0
        self._vec = vec
