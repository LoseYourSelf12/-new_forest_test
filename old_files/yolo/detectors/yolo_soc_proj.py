# Описание детектора 
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_soc_proj.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_soc_proj.zip",
    "description": "Указывает на то, что изображение является социальной рекламой",    
    "class" : "10.01 Социальная реклама",
    "classes":{
        "soc_project": "Указывает, чыто на изображении элемент, характерный для социальных проектов России",
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

    result = model.train(data=data_yaml_path, epochs=600, patience=150, batch=177, imgsz=640, lr=0.01, optimizer='SGD', seed=0, conf=0.2, name='yolo_soc_proj_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}