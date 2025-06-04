from typing import Dict, Any
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


from base import Mixer
from utils import Config


config = Config()


class RFRScikit(Mixer):
    def __init__(self, mixer_name:str):
        super().__init__()
        self.__regr = None
        self._load(mixer_name)
        self.__mixer_name = mixer_name

    def _path(self, name:str)->str:
        local_weights_dir = './mixer/weights/'
        os.makedirs(local_weights_dir, exist_ok=True)
        return os.path.join(local_weights_dir, name + '.joblib')

    def _load(self, name:str)->None:
        file_path = self._path(name)
        if os.path.isfile(file_path):
            self.__regr = joblib.load(file_path)
        else:
            self.__regr = None

    def predict(self, vec: np.array) -> float:
        return self.__regr.predict(vec)
    
    def fit(self, X: np.array, Y:np.array, property: Dict[str,Any])->None:
        if not self.__regr:
            self.__regr = RandomForestRegressor(**property)
        self.__regr.fit(X,Y)

    def save(self, name:str):
        joblib.dump(self.__regr, self._path(name))
        
    @property
    def model(self):
        return self.__regr
    
    @property
    def name(self):
        return self.__mixer_name