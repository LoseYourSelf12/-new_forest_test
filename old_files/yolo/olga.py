from typing import Dict, Any
import numpy as np
from .yolodtc import YOLODetector


class Kat_02_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="02.01 Предвыборная Агитация", 
            stopvec=np.array([1]), 
            names=['party_logo'],
            detectors = ['yolo_party_logo']
        )

        
class Kat_02_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="02.02 Иная политическая реклама", 
            stopvec=np.array([1]), 
            names=['party_logo'],
            detectors = ['yolo_party_logo'] 
        )
        

class Kat_02_05(YOLODetector):
    def __init__(self):
        super().__init__(
            name="02.05 Адвокаты", 
            stopvec=np.array([1, 1]), 
            names=['themis', 'gavel'],
            detectors = ['yolo_lawyers'] 
        )
        

class Kat_05_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.03 Информационная продукция", 
            stopvec=np.array([1]), 
            names=['inform_sign'],
            detectors = ['yolo_inform_sign'] 
        )
        

class Kat_09_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="09.02 Иные акции", 
            stopvec=np.array([1]), 
            names=['promo_ind'],
            detectors = ['yolo_promo_ind'] 
        )
        

class Kat_28_07(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.07 Строительство (ДДУ)", 
            stopvec=np.array([1]), 
            names=['real_estate'],
            detectors = ['yolo_real_estate'] 
        )
       

class Kat_28_08(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.08 Застройщик", 
            stopvec=np.array([0, 1]), 
            names=['real_estate', 'developer_logo'],
            detectors = ['yolo_real_estate', 'yolo_developer_logo'] 
        )
