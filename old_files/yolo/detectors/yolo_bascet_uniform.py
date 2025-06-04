# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_bascet_uniform.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_bascet_uniform.zip",
    "description": "Распознает баскетбольную форму",    
    "class" : ["27.01 Спорт + букмекер (основанные на риске игры, пари (азартные игры, букмекерские конторы и т.д.))"], 
        "classes":{
        "basketball_uniform": "баскетбольная форма",
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

    result = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=16, patience=30, name='yolo_bascet_uniform_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}