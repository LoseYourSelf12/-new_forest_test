from typing import Dict, Any
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import re

import base
from yolo import YOLODetector, yolo
from ocr import OcrDetector
import mixer
from utils import *
from qwen import QwenDetector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from clip import get_clip_features, init_clip, load_qwen_txt
#----------------------------------------------------------------------------------------------
# создаем детекторы
#----------------------------------------------------------------------------------------------
CTGR_NAME = ["28.08 Застройщик"]
# CSV_PATH_28_08 = r'results_csv\28.08 Застройщик_qwen.csv'
MIXER_NAME = 'mix_28_08_clip'

class Kat_28_08(YOLODetector):
    def __init__(self):
        super().__init__(
            name=CTGR_NAME[0], 
            stopvec=np.array([1]), 
            names=['developer_logo'],
            detectors = ['yolo_developer_logo'] 
        )


# текстовые детекторы пример
class MyOcrDetector_28_08(OcrDetector):
    def __init__(self):
        super().__init__(
            name=CTGR_NAME[0], 
            viptexts=[
                "застройщик", 
                "специализированный застройщик", 
                "застройщик ОOО"
            ], 
            texts=[

            ])

class QwenDetector_28_08(QwenDetector):
    def __init__(self):
        keywords_qwen = [["Реклама застройщика", 
                          'Реклама специализированных строительных услуг',
                          'Реклама жилищного строительства',
                          'Информация о строительной компании',
                          'Застройщик',
                          'Реклама инвестиционно-строительной компании',
                          'Реклама специализированных застройщиков'
                          ],
                         [r'\bзастройщ\w*\b',
                          r'\bспециализирован\w*\b',
                          ]
                         ]
        qwen_results_file = '/home/tbdbj/forest_test/qwen/qwen_ds_russ_check_3_.csv'
        super().__init__(name=CTGR_NAME[0], keywords=keywords_qwen, qwen_file=qwen_results_file, new_delimiter=';')

# Негативные детекторы
class Kat_28_08_neg(YOLODetector):
    def __init__(self):
        super().__init__(
            name='28.07 Строительство (ДДУ)', 
            stopvec=np.array([0]), 
            names=['real_estate'],
            detectors = ['yolo_real_estate'] 
        )
        
# текстовые детекторы 
class MyOcrDetector_28_08_neg(OcrDetector):
    def __init__(self):
        super().__init__(
            name='28.07 Строительство (ДДУ)', 
            viptexts=[
            ], 
            texts=[
                "жилой квартал", 
                "жилой комплекс",
                "жилой район",
                "долевое участие",
                "долевое строительство",
                "квартиры",
                "проектная декларация",
                "нашдомрф"
            ])

class QwenDetector_28_08_neg(QwenDetector):
    def __init__(self):
        keywords_qwen = [
                            ["недвижимость", "реклама недвижимости"],
                            ["проектная декларация", "проектные декларации"],
                            ["строительство"],
                            ["рассрочка"],
                            ["жилые дома", "жилье"],
                            [r"\bквартир\w*\b"]
                        ]
        qwen_results_file = '/home/tbdbj/forest_test/qwen/qwen_ds_russ_check_3_.csv'
        super().__init__(name='28.07 Строительство (ДДУ)', keywords=keywords_qwen, qwen_file=qwen_results_file, new_delimiter=';')
 

test_ctgr = base.Category(
     name=CTGR_NAME[0], 
     detectors=[Kat_28_08(), MyOcrDetector_28_08(), QwenDetector_28_08(), Kat_28_08_neg(), MyOcrDetector_28_08_neg(), QwenDetector_28_08_neg()],
     mixer=mixer.RFRScikit(MIXER_NAME)
)
#----------------------------------------------------------------------------------------------
# тест категории
#----------------------------------------------------------------------------------------------

# load_diff_files()

df_res = pd.DataFrame()

def ds_russ2024y_russ2500():
    combined = ds_combine(
        ds_russ2024y,
        ds_russ2500
    )
    for row in combined:
        yield row
        
POS_PROMPTS = ["реклама застройщика", 
                "застройщик", 
                "застройщик ООО", 
                "застройщик номер 1", 
                "специализированный застройщик",
                "реклама специализированного застройщика"
                ]

NEG_PROMPTS = ["ключи день покупки", 
               "выдаем ключи", 
               "продажа готовых домов", 
               "продажа готовых квартир", 
               "квартиры ключами",
               "квартиры отделкой",
               "готовая отделка",
               "старт продаж",
               "рассрочка застройщика",
               "аренда",
               "договор долевого участия", 
               "договор ДДУ",
               "долевое строительство",
               "строительство договор",
               "строим вместе",
               "строим",
               "строящийся объект",
               "архитектурный проект",
               "кооператив",
               "Жилой квартал",
                "жилой комплекс",
                "жилой район",
                "долевое участие",
                "квартиры",
                "проектная декларация",
                "нашдомрф"]

CORE_PHRASES  = [...]
AUX_PHRASES   = [...] 
clip_model, clip_preprocess, clip_tokenizer = init_clip()  

# ds_check_ctgr(CTGR_NAME, ds_russ2024y_russ2500)
def ds_com():
    combined = ds_combine(
        ds_russ_check_1,
    )
    for row in combined:
        yield row
for file_name, y, file_path in ds_check_ctgr(CTGR_NAME, ds_com):
    try:
        current_file_name = os.path.basename(file_path)

        local = calc_local_memory(file_path)
        
        if local == None:
            continue
        print('Обработка изображения: ', file_name)

        vec = test_ctgr.calc_vec(local)
        # pred = test_ctgr.predict(local)
        
        qwen_txt = load_qwen_txt(local, qwen_file='/home/tbdbj/forest_test/qwen/qwen_ds_russ_check_3_.csv')
        clip_vec = get_clip_features(
        text        = qwen_txt,
        model       = clip_model,
        preprocess  = clip_preprocess,
        tokenizer   = clip_tokenizer,
        pos_prompts = POS_PROMPTS,
        neg_prompts = NEG_PROMPTS,
        # core_phrases= CORE_PHRASES,
        # aux_phrases = AUX_PHRASES,
        # include_text_flags=True
        )
        pred2 = test_ctgr.predict2(local, clip_vec=clip_vec)
        # vec = np.concatenate([vec, clip_vec])

        row = pd.DataFrame([{
            'file_name': file_name,  # Оригинальное имя файла
            'current_file_name': current_file_name,  # Текущее имя файла
            'category_present': y,
            'category': y,
            'threshold_passed': pred2,
            # 'predict_threshold': pred,
            'predict_forest': pred2,
            # 'detection_vector': vec,
        }])
        
        df_res = pd.concat([df_res, row], ignore_index=True)
    except Exception as e:
        print('!!!', e)
        continue
  
CSV_PATH = f'res_and_4/{safe_filename(CTGR_NAME)}_ds_check_1.csv'

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# df_res['detection_vector'] = df_res['detection_vector'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
df_res.to_csv(CSV_PATH,  index=False, encoding='utf-8', sep=';')

df = load_df_from_csv(CSV_PATH, sep=';')

metrix = print_metrix(df=df, threshold=0.45)
report(test_ctgr=test_ctgr, metrics=metrix, importance=None, dataset='ds_russ_check_1', pref='res_and_4', suf='_28_08_ds_check_1')
#----------------------------------------------------------------------------------------------
# тест обучения
#----------------------------------------------------------------------------------------------
X, Y = load_X_Y_from_csv(test_ctgr, df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

# test_ctgr.mixer.fit(X_train, Y_train, property={
#     'n_estimators': 100, 
#     'max_depth': 50,
#     'min_samples_split': 10,
#     'min_samples_leaf': 5,  
#     'random_state': 42    
# })

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [20, 30, 50],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [3, 5, 8]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_
print("Лучшие параметры:", best_params)

# --- Обновляем test_ctgr с лучшими параметрами ---
test_ctgr.mixer.fit(X_train, Y_train, property={
    **best_params,
    'random_state': 42
})


test_ctgr.mixer.save(MIXER_NAME)

Y_pred = test_ctgr.mixer.predict(X_test)

metrix = print_metrix(Y_t=Y_test, Y_p=Y_pred, threshold=0.45)

feature_importances = feature_importances_calc(test_ctgr.mixer, X_train)

report(test_ctgr=test_ctgr, metrics=metrix, importance=feature_importances, dataset='ds_russ2024y+ds_russ2500', pref='res_and_2', suf='_28_08_ds_3_forest')
