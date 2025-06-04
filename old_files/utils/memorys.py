from typing import Callable, Any
import pandas as pd
import hashlib
import os


class Memoryes:
    def __init__(self):
        self.__hash_file = os.path.abspath('/home/tbdbj/forest_test/ocr/ocr_hash/ocr_hash.csv')
        self.__hash = self.load_hash()

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def load_hash(self):
        if os.path.exists(self.__hash_file):
            df = pd.read_csv(self.__hash_file, encoding='utf-8')
            return {row["hash"]: row["txt"] for _, row in df.iterrows()}
        return {}

    def save_hash(self):
        df = pd.DataFrame(self.__hash.items(), columns=["hash", "txt"])
        df.to_csv(self.__hash_file, index=False, encoding='utf-8')

    def load(self, key: str):
        key_hash = self._hash_key(key)
        return self.__hash.get(key_hash)

    def append(self, key: Any, value: str):
        key_hash = self._hash_key(key)
        if key_hash not in self.__hash:
            self.__hash[key_hash] = value
            self.save_hash()