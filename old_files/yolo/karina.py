from typing import Dict, Any
import numpy as np
from .yolodtc import YOLODetector


class Kat_25_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="25.02 Детское питание", 
            stopvec=np.array([1]),
            names=['baby_meal'],
            detectors = ['yolo_baby_meal']
        )

        
class Kat_05_04(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.04 Запрещенные информ. ресурсы", 
            stopvec=np.array([1]),
            names=['banned'],
            detectors = ['yolo_restricted_information_resources']
        )        

class Kat_08_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="08.01 Дистанционные продажи", 
            stopvec=np.array([1]),
            names=['distance_sales'],
            detectors = ['yolo_distance_sales']
        )   
        
class Kat_28_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.02 Кредит/Ипотека", 
            stopvec=np.array([1]),
            names=['credit_or_mortage'],
            detectors = ['yolo_credit']
        )            

class Kat_28_11(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.11 Земельные участки", 
            stopvec=np.array([0]),
            names=['land_plots'],
            detectors = ['yolo_land_plots']
        ) 

class Kat_28_12(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.12 Рассрочка", 
            stopvec=np.array([1]),
            names=['promo_ind'],
            detectors = ['yolo_promo_ind']
        )               
        
