# Описание детектора 
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_helmet_and_epaulets.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_helmet_and_epaulets.zip",  
    "description": "Указывает на то, что изображение является социальной рекламой",    
    "class" : "10.01 Социальная реклама",
    "classes":{
        "army_helmet_epaulets": "Распознает на картинке изображение армейского шлема или погон",
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

    result = model.train(data=data_yaml_path, epochs=600, patience=150, batch=64, imgsz=640, lr=0.001, optimizer='SGD', seed=0, conf=0.2, name='yolo_helmet_and_epaulets_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}