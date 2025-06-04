metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_condoms.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_condoms.zip",
    "description": "Распознает на изображении медицинские изделия",    
    "class" : "24.02 Медицинские изделия",
    "classes":{
        "condom": "Распознает на изображении укаковки и блискеры с презервативами, а также символическое и явное изображение презерватива",
    }  
}

def predict(image, local_weight_path):
    model = YOLO(local_weight_path)
    
    results = model.predict(image, verbose=False)
    
    return results
    
    
def train(data_yaml_path: str):
    model = YOLO('yolov8m')

    result = model.train(data=data_yaml_path, epochs=600, patience=150, batch=16, imgsz=640, optimizer='SGD', lr0=0.001, seed=0, conf=0.4, name='yolo_condoms')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}