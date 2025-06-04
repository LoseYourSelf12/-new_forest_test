from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from utils.config import name_id_dict


class Detector(ABC):
    """
    Базовый абстрактный класс детектора.

    Параметры конструктора:
    ---------------
    name : str
        Строка вида "01.01 Название категории" (первые 5 символов — код категории).
    features : List[str]
        Список признаков (имен признаков), по длине которого строится вектор __vec.
    """

    def __init__(self, name: str, features: List[str]) -> None:
        self._name: str = name

        key = name[:5]
        if key not in name_id_dict:
            raise ValueError(f"Категория с кодом '{key}' не найдена в name_id_dict.")
        
        self._id: str = name_id_dict[key]
        self._features: List[str] = features.copy()
        self._vec: np.ndarray = np.zeros(len(self._features), dtype=float)

    @property
    def name(self) -> str:
        """Возвращает полное имя."""
        return self._name

    @property
    def identifier(self) -> str:
        """Возвращает внутренний идентификатор (берётся из name_id_dict)."""
        return self._id

    @property
    def features(self) -> List[str]:
        """Возвращает список признаков."""
        return self._features.copy()

    @property
    def vec(self) -> np.ndarray:
        """Возвращает текущий вектор."""
        return self._vec

    def clear_vector(self) -> None:
        """
        Обнуляет все элементы вектора self._vec.
        Используется перед повторным заполнением.
        """
        self._vec[:] = 0.0

    @abstractmethod
    def __call__(self, local: Dict[str, Any]) -> None:
        """
        Абстрактный метод для вычисления/заполнения self._vec на основе данных local.
        Должен заполнять self._vec.
        Аргумент:
            local : Dict[str, Any]
                Локальные данные, необходимые для работы детектора.
        """
        ... 
