from typing import Dict, Any
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

import base
from yolo import YOLODetector, yolo
from ocr import OcrDetector, Reader
import mixer
from utils import *


#----------------------------------------------------------------------------------------------
# создаем детекторы
#----------------------------------------------------------------------------------------------
CTGR_NAME = ["05.05 Физическое лицо"]
MIXER_NAME = 'mix_05_05'  # Меняем имя микшера, должно совпадать с номеро категории

# yolo детекторы пример
class Kat_24_04(YOLODetector):
    def __init__(self):
        super().__init__( 
            name=CTGR_NAME[0], 
            stopvec=np.array([1]),
            names=['yolo_yoga_multi'],
            detectors = ['yolo_yoga_multi']
        )

    def __call__(self, local: Dict[str, Any], precheck: bool = True, threshold: float = 0.86):
        max_conf = 0.0  # Начальное значение

        for detector in self.detectors:
            results = yolo(detector, local['img'])
            boxes = results[0].boxes

            if boxes is not None and boxes.conf is not None:
                conf = boxes.conf.cpu().numpy()
                if conf.size > 0:
                    max_conf = max(max_conf, np.max(conf))

        self._vec[0] = max_conf


class Kat_05_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.05 Физическое лицо",
            stopvec=np.array([1]), 
            names=['inform_sign'],
            detectors = ['yolo_inform_sign'] 
        )


# текстовые детекторы пример
class MyOcrDetector(OcrDetector):
    def __init__(self):
        super().__init__(
            name=CTGR_NAME[0], 
            viptexts=[
                "физическое лицо", "лицо"
            ], 
            texts=[

            ])

ocr_det = MyOcrDetector()


test_ctgr = base.Category(
     name=CTGR_NAME[0], 
     detectors=[Kat_05_03(), ocr_det],
     mixer=mixer.RFRScikit(MIXER_NAME)
)
#----------------------------------------------------------------------------------------------
# тест категории
#----------------------------------------------------------------------------------------------
# Пример генераторов датасетов
# Возвращает кортеж (row['file_name'], row['category']) всех строк из разметки
# ds_wb1000  # Датасет WB_1000
# ds_russ9200  # Датасет Russ_9200
# ds_russ2500  # Датасет Russ_2500
# # Объединение датасетов в 1. Возвращает кортеж
# ds_combine(ds_russ9200, ds_wb1000)

# # Проверка наличия категории у баннера согласно разметке (file_name, y)
# ds_check_ctgr(CTGR_NAME, ds_russ9200)

# # Уменьшение количества строк 
# ds_num_iter(100, ds_wb1000) # Вернет только 10 первых строк датасета

# Пример использования правок для датасета
load_diff_files()

df_res = pd.DataFrame()

for file_name, y, file_path in ds_check_ctgr(CTGR_NAME, ds_russ_apr_24):
    try:
        current_file_name = os.path.basename(file_path)

        local = calc_local_memory(file_path)
        
        if local == None:
            continue
        print('Обработка изображения: ', file_name)

        vec = test_ctgr.calc_vec(local)

        pred = test_ctgr.predict(local)
        
        if np.max(vec) >= 0.86:
            threshold_passed = 1.0
        else:
            threshold_passed = 0.0
        row = pd.DataFrame([{
            'file_name': file_name,  # Оригинальное имя файла
            'current_file_name': current_file_name,  # Текущее имя файла
            'category_present': y,
            'threshold_passed': threshold_passed,
            'pred': pred,
            'detection_vector': vec,
            'image_text': local['txt']
        }])
        
        df_res = pd.concat([df_res, row], ignore_index=True)
    except Exception as e:
        print('!!!', e)
        continue


if df_res['category_present'].apply(lambda x: isinstance(x, list)).any():
    df_res = split_categories_to_columns(df_res, CTGR_NAME)


CSV_PATH = f'results_csv/{safe_filename(CTGR_NAME)}.csv'

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

df_res['detection_vector'] = df_res['detection_vector'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
df_res.to_csv(CSV_PATH,  index=False, encoding='utf-8')

df = load_df_from_csv(CSV_PATH)

print_metrix(df=df)

#----------------------------------------------------------------------------------------------
# тест обучения
#----------------------------------------------------------------------------------------------
df = load_df_from_csv(CSV_PATH)
X, Y = load_X_Y_from_csv(test_ctgr, df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


test_ctgr.mixer.fit(X_train, Y_train, property={
    'n_estimators': 100, 
    'max_depth': 50,
    'min_samples_split': 10,
    'min_samples_leaf': 5,  
    'random_state': 42    
})

test_ctgr.mixer.save(MIXER_NAME)

Y_pred = test_ctgr.mixer.predict(X_test)

print_metrix(Y_t=Y_test, Y_p=Y_pred)

feature_importances, feature_names = feature_importances_calc(test_ctgr.mixer, X_train)
