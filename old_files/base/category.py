from typing import List, Dict, Any
import numpy as np
from .detector import Detector
from .mixer import Mixer
from .nameiddict import name_id_dict
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


class BaseCategory:
    def __init__(self, detectors: List[Detector], mixer: Mixer = None):
        self.__dlist = detectors
        self.mixer = mixer

    @property
    def list(self) -> List[Detector]:
        return self.__dlist
    
    @property
    def stopvec(self) -> np.array:
        vec = None
        for detector in self.__dlist:
            if vec is None:
                vec = detector.stopvec
            else:
                vec = np.hstack(vec, detector.stopvec)
        return vec

    def calc_vec(self, local: Dict[str, Any]) -> np.array:
        vec = None
        for detector in self.__dlist:
            detector(local)
            if vec is None:
                vec = detector.vec
            else:
                vec = np.hstack([vec, detector.vec])
        return vec

    def predict(self, local: Dict[str, Any], precheck: bool = True, threshold: float = 0.86) -> float:
        vec = None
        values = []
        for detector in self.__dlist:
            detector(local)
            if precheck:
                values.append(np.max(detector.vec * detector.stopvec))
            if vec is None:
                vec = detector.vec
            else:
                vec = np.hstack([vec, detector.vec])
        max_value = np.max(values)
        if max_value >= threshold:
            return max_value
        vec = vec.reshape(1, -1)
        if self.mixer.model is None:
            return 0
        return self.mixer.predict(vec)[0]


class Category(BaseCategory):

    ALL: Dict[id, 'Category'] = {}

    def __init__(self, name: str, detectors: List[Detector], mixer: Mixer):
        id = name_id_dict[name]
        # for i, detector in enumerate(detectors):
        #     if detector.id != id or detector.name != name:
        #         raise Exception(
        #             f'Категория ({id}:{name}) несовместный детектор ({type(detector)} {detector.id}:{detector.name})'
        #         )
        super().__init__(
            detectors=detectors,
            mixer=mixer
        )

        self.id = name_id_dict[name]
        self.name = name

        if id in Category.ALL:
            prename = Category.ALL[id]
            raise Exception(f'Категория с номером {id} уже существует.\nИмя добовляемой категории:\n{name}\nИмя существуещей катгории:\n{prename}\n')
        Category.ALL[id] = self
