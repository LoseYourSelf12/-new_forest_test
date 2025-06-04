import os
import pandas as pd
import hashlib

class Memory:
    """Simple key->text cache stored in a CSV file."""
    def __init__(self, csv_path: str) -> None:
        self.csv_path = os.path.abspath(csv_path)
        self._cache = self._load()

    def _hash(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def _load(self) -> dict[str, str]:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path, encoding="utf-8")
            return {str(row["hash"]): str(row["txt"]) for _, row in df.iterrows()}
        return {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        df = pd.DataFrame(self._cache.items(), columns=["hash", "txt"])
        df.to_csv(self.csv_path, index=False, encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self._cache.get(self._hash(key))

    def add(self, key: str, value: str) -> None:
        h = self._hash(key)
        if h not in self._cache:
            self._cache[h] = value
            self.save()
