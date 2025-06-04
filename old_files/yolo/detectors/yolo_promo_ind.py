# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_promo_ind.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_promo_ind.zip",
    "description": "Распознает некоторые элементы акционных баннеров",
    "class" : ["09.02 Иные акции", 
               "28.12 Рассрочка"],
    "classes":{
        "promo_ind": "Символ %, зачеркнутая цена, символ денежного знака"
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

    result = model.train(data=data_yaml_path, epochs=200, imgsz=640, batch=16, patience=30, name='yolo_promo_ind_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
