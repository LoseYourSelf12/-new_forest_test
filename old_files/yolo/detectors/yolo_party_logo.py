# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_party_logo.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_party_logo.zip",
    "description": "Распознает некоторые логотипы официально зарегестрированных политических партий России (ЕдРо, КПРФ)",
    "class" : ["02.01 Предвыборная агитация", "02.02 Иная политическая реклама"],
    "classes":{
        "party_logo": "Логотипы партий ЕдРо и КПРФ"
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

    result = model.train(data=data_yaml_path, epochs=250, imgsz=640, batch=32, patience=30, name='yolo_party_logo_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
