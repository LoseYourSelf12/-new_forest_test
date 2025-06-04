# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_non_alcohol_lable.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_non_alcohol_lable.zip",
    "description": "Распознает лейбл безалкогольной продукции",    
    "class" : ["21.04 Безалкогольное пиво/вино"],
    "classes":{
        "non_alc_lable": "Лейбл безалкогольной продукции",
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

    result = model.train(data=data_yaml_path, epochs=500, imgsz=1280, batch=8, patience=50, name='yolo_non_alcohol_lable_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
