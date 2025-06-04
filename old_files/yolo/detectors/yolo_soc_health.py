metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_soc_health.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_soc_health.zip",   
    "description": "Указывает на то, что изображение является социальной рекламой",    
    "class" : "10.01 Социальная реклама",
    "classes":{
        "soc_health": "Распознает на картинке символ и эмблему проектов мэра Москвы, связанных со словом здоровье",
    }  
}

def predict(image, local_weight_path):
    model = YOLO(local_weight_path)
    
    results = model.predict(image, verbose=False)
    
    return results
    
    
def train(data_yaml_path: str):
    model = YOLO('yolov8m')

    result = model.train(data=data_yaml_path, epochs=600, patience=150, batch=16, imgsz=640, optimizer='SGD', lr0=0.001, seed=0, conf=0.4, name='yolo_soc_health_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}