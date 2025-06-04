import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Dict
from .base import Mixer


class RFRScikit(Mixer):
    """
    Реализация микшера на базе sklearn.ensemble.RandomForestRegressor.
    Загружает/сохраняет модель через joblib.
    """

    def __init__(self, mixer_name: str) -> None:
        super().__init__()
        self.__regr = None
        self.__mixer_name = mixer_name
        self._load(mixer_name)

    def _path(self, name: str) -> str:
        """
        Возвращает путь до файла с весами. Создаёт директорию, если нужно.
        """
        local_weights_dir = "./mixer/weights/"
        os.makedirs(local_weights_dir, exist_ok=True)
        return os.path.join(local_weights_dir, name + ".joblib")

    def _load(self, name: str) -> None:
        """
        Если для заданного mixer_name есть сохранённый файл, подгружаем его.
        """
        file_path = self._path(name)
        if os.path.isfile(file_path):
            self.__regr = joblib.load(file_path)
        else:
            self.__regr = None

    def predict(self, vec: np.ndarray) -> float:
        """
        Предсказывает по одному вектору vec (shape=(n_features,)).
        """
        if self.__regr is None:
            raise RuntimeError(f"Модель '{self.__mixer_name}' не загружена и не обучена.")
        # sklearn ожидает двумерный массив (n_samples, n_features), поэтому reshape
        vec_2d = vec.reshape(1, -1)
        return float(self.__regr.predict(vec_2d)[0])

    def fit(self, X: np.ndarray, Y: np.ndarray, properties: Dict[str, Any]) -> None:
        """
        Обучает RandomForestRegressor. 
        В properties можно передать гиперпараметры для RandomForestRegressor.
        """
        if self.__regr is None:
            self.__regr = RandomForestRegressor(**properties)
        self.__regr.fit(X, Y)

    def save(self, name: str) -> None:
        """
        Сохраняет текущую модель под именем name.joblib.
        """
        if self.__regr is None:
            raise RuntimeError("Нет модели для сохранения.")
        joblib.dump(self.__regr, self._path(name))

    @property
    def model(self):
        """Возвращает внутренний sklearn-регрессор (RandomForestRegressor)."""
        return self.__regr

    @property
    def name(self) -> str:
        """Имя этого микшера (используется при сохранении/загрузке)."""
        return self.__mixer_name
