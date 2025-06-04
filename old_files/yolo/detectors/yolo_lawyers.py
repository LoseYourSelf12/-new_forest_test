# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_lawyers.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_lawyers.zip",
    "description": "Распознает некоторые символы правосудия (статуя Фемиды, судейский молоток)",
    "class" : "02.05 Адвокаты",
    "classes":{
        "themis": "Cтатуя богини правосудия, с повязкой на глазах, с весами в одной руке и опущенным мечом в другой",
        "gavel": "Молоток судьи, расположенный на подставке или находящийся в руках"
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

    result = model.train(data=data_yaml_path, epochs=200, imgsz=320, batch=32, patience=30, name='yolo_lawyers_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}