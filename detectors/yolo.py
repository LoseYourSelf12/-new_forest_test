from __future__ import annotations
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
from .base import Detector

# Simple cache for loaded YOLO models
_MODEL_CACHE: Dict[str, YOLO] = {}

# Directory with weight files
LOCAL_WEIGHTS_DIR = os.path.join("yolo", "weights")


def _load_model(name: str) -> YOLO:
    path = os.path.join(LOCAL_WEIGHTS_DIR, name + ".pt")
    if path not in _MODEL_CACHE:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Weight file not found: {path}")
        _MODEL_CACHE[path] = YOLO(path)
    return _MODEL_CACHE[path]


def _yolo_predict(model_name: str, image: Image.Image):
    model = _load_model(model_name)
    return model.predict(image, verbose=False)


class YOLODetector(Detector):
    """Detector based on YOLO models."""

    def __init__(self, name: str, classes: List[str], detectors: List[str]):
        super().__init__(name=name, features=classes)
        self._detectors = detectors

    def __call__(self, local: Dict[str, Any]) -> None:
        if "img" not in local:
            raise ValueError("local must contain 'img' with an image")
        global_map = {c: i for i, c in enumerate(self._features)}
        conf_vector = np.zeros(len(self._features), dtype=float)
        image = local["img"]
        for det_name in self._detectors:
            results = _yolo_predict(det_name, image)
            boxes = results[0].boxes
            if boxes is None or boxes.conf is None:
                continue
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for c, confidence in zip(cls, conf):
                class_name = results[0].names[c]
                if class_name in global_map:
                    idx = global_map[class_name]
                    conf_vector[idx] = max(conf_vector[idx], float(confidence))
        self._vec = conf_vector
