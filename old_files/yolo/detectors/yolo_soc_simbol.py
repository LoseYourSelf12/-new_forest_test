# Описание детектора 
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_soc_simbol.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_soc_simbol.zip",    
    "description": "Указывает на то, что изображение является социальной рекламой",    
    "class" : "10.01 Социальная реклама",
    "classes":{
        "soc_simbol": "Указывает, что на изображении символ социального или госоргана (гибдд, герба Москвы или СПб), символа соц.рекламы",
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

    result = model.train(data=data_yaml_path, epochs=500, patience=50, batch=22, imgsz=864, lr=0.005, optimizer='SGD', seed=0, conf=0.4, name='yolo_soc_simbol_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}