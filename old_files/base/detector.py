from typing import List, Dict, Any, Callable
import numpy as np
import os
from .nameiddict import name_id_dict


class Detector:
    def __init__(self, ctgrname: str, stopvec: np.array, names: List[str]):
        self.__id = name_id_dict[ctgrname]
        self.__name = ctgrname
        self.__stopvec = stopvec
        self._vec = np.full_like(self.__stopvec, 0.0, dtype=float)
        self.names = names

    @property
    def stopvec(self)->np.array:
        return self.__stopvec
    
    @property
    def vec(self)->np.array:
        return self._vec

    @property
    def id(self) -> int:
        return self.__id

    @property
    def name(self) -> str:
        return self.__name
    
    def clear_vec(self):
        self._vec = np.full_like(self.__stopvec, 0.0, dtype=float)

    def __call__(self, local: Dict[str, Any]):
        """Расчет вектора признаков детектора"""
        raise NotImplementedError
    

class TextDetector(Detector):
    def __init__(self, ctgrname: str, vip_texts: List[str], texts: List[str], tag:str, cmpfnc:Callable):
        super().__init__(
            ctgrname=ctgrname,
            stopvec=np.array([1]*len(vip_texts)+[0]*len(texts)),
            names=vip_texts + texts
        )
        self.__tag = tag
        self.__cmpfnc = cmpfnc

    def __call__(self, local:Dict[str, Any]):
        self.clear_vec()
        txt = local[self.__tag]
        for i, text in enumerate(self.names):
            stop = self.stopvec[i]
            if self.__cmpfnc(text, txt):
                self._vec[i] = 1.0
                if stop == 1:
                    return


class FileTextDetector(TextDetector):
    def __init__(self, ctgrname: str, filename: str, tag: str, cmpfnc: Callable):
        ocr_texts_dir = "./texts/"
        os.makedirs(ocr_texts_dir, exist_ok=True)
        ctgr_find = False
        vip_texts = []
        texts = []
        file_path = ocr_texts_dir + filename
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line == '' or line[0] == '*':
                    continue
                if line[0] == '[':
                    if ctgr_find:
                        break
                    file_ctgrname = line[1:-1].strip()
                    if file_ctgrname == ctgrname:
                        ctgr_find = True
                        continue
                else:
                    if ctgr_find:
                        if line[0] == '#':
                            vip_texts.append((line[1:].strip()).upper())
                        else:
                            texts.append(line.strip().upper())

        super().__init__(
            ctgrname=ctgrname,
            vip_texts=vip_texts,
            texts=texts,
            tag=tag,
            cmpfnc=cmpfnc
        )


class QwenDetector(FileTextDetector):
    def __init__(self, filename: str, name: str):
        super().__init__(
            filename=filename, ctgrname=name,
            tag='txt', cmpfnc=()# поправить отсылку на ocr
        )
