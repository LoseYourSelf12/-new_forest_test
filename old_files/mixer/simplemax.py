import base
import numpy as np


class SimpleMax(base.Mixer):
    def __init__(self):
        super().__init__()

    def predict(self, vec: np.array) -> float:
        return float(vec.max())
    