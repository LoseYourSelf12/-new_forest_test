from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import easyocr
from python_Levenshtein import distance  # rename from Levenshtein
from .base import Detector


def fuzzy_check(phrase: str, text: str) -> bool:
    phrase_len = len(phrase.split())
    words = text.split()
    text_len = len(words)
    windows = [' '.join(words[i:i + phrase_len]) for i in range(max(text_len - phrase_len + 1, 0))]
    levdist = [[w, distance(phrase, w)] for w in windows]
    threshold = 1 + len(phrase) // 3
    result = [rt for _, rt in levdist if rt < threshold]
    if not result and len(set(phrase.split())) > 1:
        # simplified long_phrase check
        min_item = min(levdist, key=lambda x: x[1]) if levdist else ['', 100]
        if distance(phrase.replace(' ', ''), min_item[0].replace(' ', '')) < 1:
            return True
        return False
    return bool(result)


class OCRDetector(Detector):
    """Detector that searches keywords in OCR text."""

    def __init__(self, name: str, texts: List[str]):
        super().__init__(name=name, features=texts)
        self._texts = [t.upper() for t in texts]
        self._reader: easyocr.Reader | None = None

    def _ensure_reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(['ru', 'en'], gpu=True)
        return self._reader

    def _extract_text(self, image: Image.Image) -> str:
        reader = self._ensure_reader()
        result = reader.readtext(np.array(image), detail=0)
        return ' '.join(result)

    def __call__(self, local: Dict[str, Any]) -> None:
        text = local.get('ocr_text') or local.get('txt')
        if text is None:
            if 'img' not in local:
                raise ValueError("local must contain 'img' or 'ocr_text'")
            text = self._extract_text(local['img'])
        text = text.upper()
        vec = np.zeros(len(self._texts), dtype=float)
        for i, phrase in enumerate(self._texts):
            if fuzzy_check(phrase, text):
                vec[i] = 1.0
        self._vec = vec
