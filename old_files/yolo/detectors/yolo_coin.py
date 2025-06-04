# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_coin.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_coin.zip",
    "description": "Распознает монеты",    
    "class" : ["28.14 Ломбарды"],
    "classes":{
        "coin": "монеты",
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

    result = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=16, patience=30, name='yolo_coin_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}