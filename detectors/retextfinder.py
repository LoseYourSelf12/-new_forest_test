from __future__ import annotations
import re
from typing import List, Iterable


class ReTextFinder:
    """Utility class to search multiple regex groups in text."""

    def __init__(self, words_list: List[List[str]]):
        self._patterns = []
        self._original_words: List[str] = []
        shift = 0
        for words in words_list:
            mapping = {}
            parts = []
            for i, word in enumerate(words):
                group = f"g{shift + i}"
                mapping[group] = shift + i
                parts.append(f"(?P<{group}>{word})")
                self._original_words.append(word)
            shift += len(words)
            pattern = re.compile("|".join(parts))
            self._patterns.append((mapping, pattern))

    def finditer(self, text: str) -> Iterable[int]:
        found = set()
        for mapping, pattern in self._patterns:
            for mo in pattern.finditer(text):
                for group_name, index in mapping.items():
                    if mo.group(group_name) is not None and index not in found:
                        found.add(index)
                        yield index

    @property
    def features(self) -> List[str]:
        return self._original_words
