# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_pharma.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_pharma.zip",
    "description": "Распознает на картинке изображение лекарственных прпепаратов",    
    "class" : "24.03 Лекарственные препараты",
    "classes":{
        "pharma": "Распознает на картинке изображение лекарственных препаратов в коробках, блистерах и капсул и таблеток без упаковки",
    }  
}

def predict(image, local_weight_path):
    # Пример для YOLO v8
    model = YOLO(local_weight_path)
    
    # Распознаем
    results = model.predict(image, verbose=False)
    
    return results
    
    
def train(data_yaml_path: str):
    model = YOLO('yolov8x')

    result = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=16, patience=30, name='yolo_pharma_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
