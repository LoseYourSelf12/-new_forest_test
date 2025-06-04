metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_med_glove.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_med_glove.zip", 
    "description": "Распознает на картинке изображение медицинских изделий",    
    "class" : "24.02 Медицинские изделия",
    "classes":{
        "glove": "Распознает на картинке изображение медицинских перчаток",
    }  
}

def predict(image, local_weight_path):
    model = YOLO(local_weight_path)
    
    results = model.predict(image, verbose=False)
    
    return results
    
    
def train(data_yaml_path: str):
    model = YOLO('yolov8m')

    result = model.train(data=data_yaml_path, epochs=600, patience=150, batch=32, imgsz=640,  name='yolo_med_glove_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}