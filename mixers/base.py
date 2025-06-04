from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class Mixer(ABC):
    """Базовый абстрактный класс микшера."""

    @abstractmethod
    def predict(self, vec: np.ndarray) -> float:
        """
        На вход даём одномерный numpy-массив vec (например, объединённый вектор всех детекторов),
        возвращаем число.
        """
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, properties: Dict[str, Any]) -> None:
        """
        Тренировка модели на выборке X, Y с учётом hyperparameters properties.
        """
        ...

    @abstractmethod
    def save(self, name: str) -> None:
        """
        Сохранить веса/состояние модели под указанным именем.
        """
        ...
