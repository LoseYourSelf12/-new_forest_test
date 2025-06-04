from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO
import utils
import base
import numpy as np
import os


LOCAL_WEIGHTS_DIR = './yolo/weights/'
LOCAL_DETECTORS_DIR = './yolo/detectors/'
DETECTORS_PATH = 'moderate/test_server/detectors/yolo/src/'
# Глобальный кэш для хранения уже загруженных моделей по пути к весам
model_cache = {}

def yolo(model_name:str, image: Image):
    local_weights_dir = LOCAL_WEIGHTS_DIR
    local_detectors_dir = LOCAL_DETECTORS_DIR
    detectors_path = DETECTORS_PATH

    # проверка наличия локально файла модели
    model_path = os.path.join(local_detectors_dir, model_name + '.py')
    if not os.path.isfile(model_path):
        # скачиваем файл модели
        hub = utils.Hub()
        s3_model_path = detectors_path + model_name + '.py'
        if hub.is_exist(s3_model_path):
            hub.download(s3_model_path, model_path)
        else:
            print(f"\n\n{s3_model_path} \t{model_path}\n\n")
            raise Exception(f"Детектора с именем {model_name} не найден ({s3_model_path})")

    # загружаем модель и получаем методанные
    model_module = utils.load_module(local_detectors_dir, model_name + '.py', {"YOLO": YOLO})

    # получаем пути весов как локальных так и в s3
    s3_weight_path = getattr(model_module, "metadata")["model_path"]
    local_weight_path = local_weights_dir + os.path.basename(s3_weight_path)

    if not os.path.isfile(local_weight_path):
        # загружаем веса с s3
        hub = utils.Hub()
        if hub.is_exist(s3_weight_path):
            hub.download(s3_weight_path, local_weight_path)
        else:
            raise Exception(f"Веса для модели {model_name} на s3 не найдены ({s3_weight_path})")

    # Берем исходную функцию predict из детектора
    original_predict = getattr(model_module, "predict")

    # Проверяем, не патчили ли мы ее
    if not hasattr(model_module, "_original_predict"):
        # Сохраняем исходную функцию на всякий случай
        model_module._original_predict = original_predict

        def cached_predict(image, local_weight_path):
            if local_weight_path not in model_cache:
                model_cache[local_weight_path] = YOLO(local_weight_path)

            model = model_cache[local_weight_path]

            return model.predict(image, verbose=False)

        # Подменяем функцию predict
        setattr(model_module, "predict", cached_predict)

    # Вызываем с подменой и кэшем
    results = model_module.predict(image, local_weight_path)
    return results


class YOLODetector(base.Detector):
    def __init__(self, name: str, stopvec: np.array, names: List[str], detectors: List[str]):
        super().__init__(
            ctgrname=name,
            stopvec=stopvec,
            names=names
        )

        assert len(stopvec) == len(names), "Длинна stopvrc и names различны"

        self.detectors = detectors

    def __call__(self, local:Dict[str, Any]):
        global_class_map = {}
        index = 0
        results_list = []
        for detector in self.detectors:
            results = yolo(detector, local['img'])
            results_list.append(results)
            for local_id, class_name in results[0].names.items():
                if class_name not in global_class_map:
                    global_class_map[class_name] = index
                    index += 1
        
        conf_vector = np.zeros(len(self.names), dtype=np.float32)

        for results in results_list:
            boxes = results[0].boxes
            if boxes is not None and boxes.conf is not None:
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for c, confidence in zip(cls, conf):
                    class_name = results[0].names[c]
                    global_id = global_class_map[class_name]
                    conf_vector[global_id] = max(conf_vector[global_id], confidence)

        self._vec = conf_vector