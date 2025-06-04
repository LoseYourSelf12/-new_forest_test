# Описание детектора
metadata = {
    "model_path": "moderate/test_server/detectors/yolo/weights/yolo_medicine.pt",
    "dataset_path": "moderate/test_server/datasets/yolo/yolo_medicine.zip",  
    "description": "Распознает на картинке изображение принадлежности к аптеке или медорганизации",    
    "class" : "24.06 МедОрганизация/Аптека",
    "classes":{
        "medicine": "Распознает на картинке изображение медицинского креста, чаши со змеем, эмблемы/логотипа аптеки, символ зуба или зубного импланта",
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

    result = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=32, patience=300, optimizer='SGD', seed=0, name='yolo_medicine_')
    return result


def get_metadata(detector_name: str):
    return {'name': detector_name, **metadata}
