from typing import Dict, Any
import numpy as np
from .yolodtc import YOLODetector


class Kat_10_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="10.01 Социальная реклама", 
            stopvec=np.array([1, 1, 1, 1, 1]),
            names=['soc_health',  'soc_simbol', 'soc_project', 'soc_army', 'army_helmet_epaulets'],
            detectors = ["yolo_soc_health",  "yolo_soc_simbol", "yolo_soc_proj", "yolo_soc_army", "yolo_helmet_and_epaulets"]
        )

        
class Kat_24_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.02 Медицинские изделия", 
            stopvec=np.array([1, 1, 1, 1, 1, 1, 1]),
            names=['glove', 'microscope', 'scalpel', 'mask', 'blood_pressure_monitor', 'stethoscope', 'condom'],
            detectors = ["yolo_med_glove", "yolo_micro", "yolo_scalpel", "yolo_mask", "yolo_blood_monit_big", "yolo_stethoscope", "yolo_condoms"]
        )
        

class Kat_24_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.03 Лекарственные препараты", 
            stopvec=np.array([1]),
            names=['pharma'],
            detectors = ["yolo_pharma"]
        )
        

class Kat_24_06(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.06 МедОрганизация/Аптека", 
            stopvec=np.array([1]),
            names=['medicine'],
            detectors = ["yolo_medicine"]
        )


class Kat_21_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.01 Алкоголь, демонстрация процесса потребления алкоголя",
            stopvec=np.array([1]),
            names=['alc_bottle'],
            detectors = ["yolo_alc_bottle"]
        ) 
        

class Kat_21_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.02 Алкомаркет",
            stopvec=np.array([1]),
            names=['alc_bottle'],
            detectors = ["yolo_alc_bottle"]
        ) 
        

class Kat_21_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.03 Бар, ресторан",
            stopvec=np.array([1]),
            names=['alc_bottle'],
            detectors = ["yolo_alc_bottle"]
        ) 
        

class Kat_59(YOLODetector):
    def __init__(self):
        super().__init__(
            name="59. Табак, табачная продукция, табачные изделия и курительные принадлежности, в том числе трубок, кальянов, сигаретная бумага, зажигалки, демонстрация процесса курения",
            stopvec=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            names=['cigarette', 'pipes', 'smoke', 'smoking', 'no-smoking', 'pack', 'lighter', 'hookah', 'cig_roll'],
            detectors = ["yolo_smokers"]
        ) 
        

class Kat_49(YOLODetector):
    def __init__(self):
        super().__init__(
            name="49. Оружие и продукция военного назначения",
            stopvec=np.array([1, 1, 1]),
            names=['gun', 'weapon_box', 'edged_weapon'],
            detectors = ["yolo_gun", "yolo_weapon_box", "yolo_edged_weapon"]
        )

        
class Kat_83(YOLODetector):
    def __init__(self):
        super().__init__(
            name="83. Казино (в т.ч. онлайн-казино)",
            stopvec=np.array([1, 1]),
            names=['playing-cards', 'chips'],
            detectors = ["yolo_playing_cards", "yolo_chips"]
        ) 