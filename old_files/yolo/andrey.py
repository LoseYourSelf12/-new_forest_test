from typing import Dict, Any
import numpy as np
from .yolodtc import YOLODetector, yolo


class Kat_24_04(YOLODetector):
    def __init__(self):
        super().__init__( 
            name="24.04 Методы народной медицины", 
            stopvec=np.array([1]),
            names=['yolo_yoga_multi'],
            detectors = ['yolo_yoga_multi']
        )

    def __call__(self, local: Dict[str, Any]):
        max_conf = 0.0  # Начальное значение

        for detector in self.detectors:
            results = yolo(detector, local['img'])
            boxes = results[0].boxes

            if boxes is not None and boxes.conf is not None:
                conf = boxes.conf.cpu().numpy()
                if conf.size > 0:
                    max_conf = max(max_conf, np.max(conf))

        self._vec[0] = max_conf


class Kat_05_07(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.07 QR-код / адрес сайта", 
            stopvec=np.array([1, 1]), 
            names=['qrcode', 'qrcode_true'],
            detectors = ['yolo_qrcode']
        )


class Kat_21_04(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.04 Безалкогольное пиво/вино", 
            stopvec=np.array([1, 0]),
            names=['non_alc_lable', 'alc_bottle'],
            detectors = ['yolo_non_alcohol_lable', 'yolo_alc_bottle']
        )


class Kat_24_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.01 Медицинские услуги", 
            stopvec=np.array([1, 0, 1, 1, 1, 1]),
            names=['dentist_items', 'object', 'syringe', 'glove', 'microscope', 'stethoscope'],
            detectors = ['yolo_dentist', 'yolo_syringe', 'yolo_med_glove', 'yolo_micro', 'yolo_stethoscope']
        )


class Kat_24_05(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.05 Методы лечения, профилактики и диагностики", 
            stopvec=np.array([1, 1, 1]),
            names=['syringe', 'microscope', 'stethoscope'],
            detectors = ['yolo_syringe', 'yolo_micro', 'yolo_stethoscope']
        )


class Kat_28_06(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.06 Инвест-платформа", 
            stopvec=np.array([1, 1, 1, 1]),
            names=['sber_inv', 'vtb_inv', 't_inv', 'alfa_inv'],
            detectors = ['yolo_invest']
        )


class Kat_29_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="29.03 Криптовалюта", 
            stopvec=np.array([1, 1, 1, 1, 1]),
            names=['BTC', 'ETH', 'DOGE', 'USDT', 'USDC'],
            detectors = ['yolo_crypto_multi']
        )
