# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_inform_sign.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_inform_sign.zip",
    "description": "Распознает информационный знак возрастного ограничения (0+, 6+, 12+, 16+, 18+)",
    "class" : "05.03 Информационная продукция",
    "classes":{
        "inform_sign": "Знак возрастного ограничения вида: 0+, 6+, 12+, 16+, 18+"
    }
}

def predict(image, local_weight_path):
    # Пример для YOLO v8
    model = YOLO(local_weight_path)
    
    # Распознаем
    results = model.predict(image, verbose=False)
    
    return results
    
    
def train(data_yaml_path: str):
    model = YOLO('yolov8m')

    result = model.train(data=data_yaml_path, epochs=200, imgsz=640, batch=16, patience=30, name='yolo_inform_sign_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
