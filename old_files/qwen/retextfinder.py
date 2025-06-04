import re
from typing import List

class ReTextFinder:
    def __init__(self, words_list: List[List[str]]):
        self.__patterns = [] 
        self.__original_words = [] 
        shift = 0
        for words in words_list:
            mapping = {}
            group_parts = []
            for i, word in enumerate(words):
                group_name = f"g{shift + i}"
                mapping[group_name] = shift + i
                group_parts.append(f"(?P<{group_name}>{word})")
                self.__original_words.append(word)
            shift += len(words)
            pattern_str = "|".join(group_parts)
            compiled = re.compile(pattern_str)
            self.__patterns.append((mapping, compiled))

    def finditer(self, text: str):
        found = set()
        for mapping, pattern in self.__patterns:
            for mo in pattern.finditer(text):
                for group_name, index in mapping.items():
                    if mo.group(group_name) is not None:
                        if index in found:
                            continue
                        found.add(index)
                        yield index

    @property
    def features(self) -> List[str]:
        return self.__original_words