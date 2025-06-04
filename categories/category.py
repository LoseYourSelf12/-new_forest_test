from typing import List, Dict, Type, Any
import numpy as np
from detectors.base import Detector
from mixers.base import Mixer


class Category:
    """
    Класс, описывающий конкретную «категорию» — совокупность детекторов и микшеров для неё.

    Параметры конструктора:
    ---------------
    name : str
        Название категории, например.
    detectors : List[Detector]
        Список уже проинициализированных детекторов (экземпляров классов, унаследованных от Detector).
    mixers : Dict[str, Type[Mixer]]
        Словарь, где ключ — имя микшера (строка), а значение — класс микшера (унаследованного от Mixer).
        Например: { "rfr_default": RFRScikit, "my_other_mixer": MyCustomMixer }
    """

    def __init__(
        self,
        name: str,
        detectors: List[Detector],
        mixers: Dict[str, Type[Mixer]],
    ) -> None:
        self._name: str = name
        self._detectors: List[Detector] = detectors.copy()
        self._mixers: Dict[str, Type[Mixer]] = mixers.copy()

    @property
    def name(self) -> str:
        """Возвращает полное имя категории."""
        return self._name

    @property
    def detectors(self) -> List[Detector]:
        """Возвращает список детекторов (экземпляров)."""
        return self._detectors.copy()

    @property
    def mixers(self) -> List[str]:
        """Возвращает список имён доступных микшеров для этой категории."""
        return list(self._mixers.keys())

    def calc_vec(self, local: Dict[str, Any]) -> np.ndarray:
        """
        Вызывает каждый детектор из self._detectors, передавая ему local,
        и конкатенирует все их вектора (np.ndarray) последовательно в один итоговый массив.
        """

        if not self._detectors:
            raise ValueError("Список детекторов пуст.")

        concatenated = []
        for det in self._detectors:
            det.clear_vector()
            det(local)
            concatenated.append(det.vec)

        # Склеиваем все под-вектора в один одномерный массив
        return np.concatenate(concatenated, axis=0)

    def predict(self, local: Dict[str, Any], mixer_name: str) -> float:
        """
        Сначала вычисляет объединённый вектор через calc_vec(local),
        затем создаёт экземпляр микшера по ключу mixer_name из self._mixers
        и вызывает у него predict(combined_vec). Возвращает значение.
        """
        if mixer_name not in self._mixers:
            raise KeyError(f"Микшер с именем '{mixer_name}' не найден в категории '{self._name}'.")

        combined_vec = self.calc_vec(local)
        mixer_cls = self._mixers[mixer_name]
        mixer_instance = mixer_cls(mixer_name)
        score = mixer_instance.predict(combined_vec)
        return score
