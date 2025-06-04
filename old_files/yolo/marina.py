from typing import Dict, Any
import numpy as np
from .yolodtc import YOLODetector

class Kat_29_02(YOLODetector):    
    def __init__(self):
        super().__init__(            
            name="29.02 Цифровые финансовые активы",
            stopvec=np.array([0,0,0,1]),            
            names=['graphics', 'bar','plot_bb','plot'],
            detectors = ['yolo_information_line','yolo_information_bar_char','yolo_circle']        
            )
        
class Kat_29_01(YOLODetector):
    def __init__(self):
        super().__init__(            
            name="29.01 Ценные бумаги",
            stopvec=np.array([0, 1,0,1]),            
            names=['bar', 'plot_bb','graphics','plot'],
            detectors = ['yolo_information_bar_char', 'yolo_information_line','yolo_circle']        
            )
        
class Kat_27_01(YOLODetector):    
    def __init__(self):
        super().__init__(            
            name="27.01 Спорт + букмекер (основанные на риске игры, пари (азартные игры, букмекерские конторы и т.д.))",            
            stopvec=np.array([1,1,0,0,1,1,0,0,1,0]),
            names=['ball', 'rim', 'ball','1xBet', 'player', 'referee', 'player_hockey_sticks', 'Boxing_Glove', 'Football', 'basketball_uniform'],
            detectors=['yolo_basket_ball', 'yolo_basket', 'yolo_ball', 'yolo_1xBet', 'yolo_player_and_referee', 'yolo_hockey_stick', 'yolo_gloves', 'yolo_football', 'yolo_bascet_uniform']
        )

class Kat_05_05(YOLODetector):
    def __init__(self):
        super().__init__(            
            name="05.05 Физическое лицо",
            stopvec=np.array([1]),            
            names=['face'],
            detectors=['yolo_face']        
            )

class Kat_29_02(YOLODetector):
    def __init__(self):
        super().__init__(            
            name="29.02 Цифровые финансовые активы",
            stopvec=np.array([0, 1, 1]),            
            names=['binance', 'MetaMask', 'trust_wallet'],
            detectors=['yolo_binance', 'yolo_metamask', 'yolo_trust_wallet']        
            )

class Kat_28_14(YOLODetector):
    def __init__(self):        
        super().__init__(
            name="28.14 Ломбарды",            
            stopvec=np.array([0, 1]),
            names=['coin', 'Jewellery'],            
            detectors=['yolo_coin', 'yolo_jewelry']
        )