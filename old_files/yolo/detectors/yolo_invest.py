# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_invest.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_invest.zip",
    "description": "Детектор распознает различные логотипы инвестиционных платформ", 
    "class" : ["28.06 Инвест-платформа"],
    "classes":{
        "sber_inv": "Сбер инвестиции",
        "vtb_inv": "ВТБ инвестиции",
        "t_inv": "Тинькофф инвестиции",
        "alfa_inv": "Альфа инвестиции"
    }  
}


def predict(image, local_weight_path):
    # Пример для YOLO v8
    model = YOLO(local_weight_path)
    
    # Распознаем
    results = model.predict(image, verbose=False)
    
    return results
    
    
def train(data_yaml_path: str):
    model = YOLO('yolov8l')

    result = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=32, patience=30, name='yolo_invest_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
