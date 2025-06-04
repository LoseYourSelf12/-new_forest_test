import numpy as np
import pandas as pd
from base.detector import Detector
from .retextfinder import ReTextFinder


class QwenDetector(Detector):
    def __init__(self, name: str, keywords: list, qwen_file: str, new_delimiter: str = ','):
        if all(isinstance(item, list) for item in keywords):
            flattened_keywords = [word for sublist in keywords for word in sublist]
            original_keywords = keywords
        else:
            flattened_keywords = keywords
            original_keywords = [keywords]
        stopvec = np.ones(len(flattened_keywords), dtype=float)
        super().__init__(ctgrname=name, stopvec=stopvec, names=flattened_keywords)
        self.qwen_file = qwen_file

        self.textfinder = ReTextFinder(original_keywords)

        self.df = pd.read_csv(self.qwen_file, encoding='utf-8', delimiter=new_delimiter)

    def __call__(self, local: dict):
        self.clear_vec()
        file_id = local.get('current_file_name', local.get('file_name'))

        cached_text = None
        if cached_text is not None:
            qwen_text = cached_text
        else:
            qwen_text = self.load_qwen_text(file_id)
            # self.cache.append(file_id, qwen_text)

        for idx in self.textfinder.finditer(qwen_text):
            self._vec[idx] = 1.0

    def load_qwen_text(self, file_id: str) -> str:
        try:
            row = self.df.loc[self.df['file_name'] == file_id]
            if not row.empty:
                return row.iloc[0]['texts']
        except Exception as e:
            print(f"Ошибка при чтении CSV файла qwen: {e}")
        return ""
