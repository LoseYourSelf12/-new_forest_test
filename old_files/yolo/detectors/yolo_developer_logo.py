# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_developer_logo.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_developer_logo.zip",
    "description": "Распознает некоторые логотипы компаний застройщиков (те, что встретились на макетах от заказчика)",
    "class" : ["28.07 Строительство (ДДУ)", 
               "28.09 Застройщик"],
    "classes":{
        "developer_logo": "Логотипы компаний застройщиков (A101, Level, Гранель и др.)"
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

    result = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=16, patience=30, name='yolo_developer_logo_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
