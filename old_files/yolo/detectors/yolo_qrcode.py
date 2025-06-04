# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_qrcode.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_qrcode.zip",
    "description": "Детектор распознает контрольные точки QR кода, а также сам QR код целиком", 
    "class" : ["5.07 QR-код / адрес сайта"],
    "classes":{
        "qrcode_true": "QR целиком",
        "qrcode": "Контрольные точки QR"
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

    result = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=32, patience=30, name='yolo_qrcode_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
