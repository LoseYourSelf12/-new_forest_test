# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_real_estate.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_real_estate.zip",
    "description": "Распознает некоторые объекты недвижимости: дома, жилые комплексы, квартиры",
    "class" : ["28.07 Строительство (ДДУ)", 
               "28.09 Застройщик", 
               "28.10 Построенная недвижимость (продажа/аренда)", 
               "28.13 Строительство (Кооператив)"],
    "classes":{
        "real_estate": "Объект недвижимости (здание, группа зданий, квартира)"
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

    result = model.train(data=data_yaml_path, epochs=250, imgsz=640, batch=16, patience=30, name='yolo_real_estate_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
