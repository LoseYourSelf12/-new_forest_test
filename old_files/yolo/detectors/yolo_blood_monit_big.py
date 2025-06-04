# Описание детектора 
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_blood_monit_big.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_blood_monit_big.zip",
    "description": "Распознает на картинке изображение медицинских изделий",    
    "class" : "24.02 Медицинские изделия",
    "classes":{
        "blood_pressure_monitor": "Распознает на картинке изображение монитора от прибора для измерения артериального давления, а также тонометра целиком",
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

    result = model.train(data=data_yaml_path, epochs=600, patience=150, batch=236, imgsz=640, lr=0.005, optimizer='SGD', seed=0, conf=0.4, name='yolo_blood_monit_big_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}