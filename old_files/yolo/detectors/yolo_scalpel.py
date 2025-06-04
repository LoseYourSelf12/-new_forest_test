# Описание детектора 
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_scalpel.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_scalpel.zip",   
    "description": "Распознает на картинке изображение медицинских изделий",    
    "class" : "24.02 Медицинские изделия",
    "classes":{
        "scalpel": "Распознает на картинке изображение медицинского скальпеля лезвия скальпеля или ручки для скальпеля",
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

    result = model.train(data=data_yaml_path, epochs=600, patience=150, batch=36, imgsz=640, lr=0.01, optimizer='SGD', seed=0, conf=0.2, name='yolo_scalpel_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}