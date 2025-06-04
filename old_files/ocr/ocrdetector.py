from typing import List
import base
from .methods import fuzzy_check

class FileOcrDetector(base.FileTextDetector):
    def __init__(self, filename: str, name: str):
        super().__init__(
            filename=filename, ctgrname=name,
            tag='txt', cmpfnc=fuzzy_check
        )

class OcrDetector(base.TextDetector):
    def __init__(self, name:str, viptexts: List[str], texts: List[str]):
        super().__init__(
            ctgrname=name,
            vip_texts=viptexts,
            texts=texts,
            tag='txt', cmpfnc=fuzzy_check
        )
