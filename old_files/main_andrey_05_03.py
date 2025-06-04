from typing import Dict, Any
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

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
CTGR_NAME = ["05.03 Информационная продукция"]
# CSV_PATH_05_03 = r'results_csv\05.03 Информационная продукция.csv'
MIXER_NAME = 'mix_05_03_clip'

class MyOcrDetector(OcrDetector):
    def __init__(self):
        super().__init__(
            name=CTGR_NAME[0], 
            viptexts=[
                "правила участия", 
                "премьера", 
                'информационные услуги', 
                'смотри в приложении',
                'телеканал',
                'премьера клипа',
                'премьера трека',
                'мьюзикл',
                'спектакль',
                'фестиваль',
                'концерт'
            ], 
            texts=[
                'театр',
                'песня',
                'трек',
                'радио',
                'пьесса',
                'опера',
                'выставка',
                'приложен',
                'программное обеспечение'
            ])

class QwenDetector(QwenDetector):
    def __init__(self):
        keywords_qwen = [["Реклама телевизионного сериала", 
                          "Реклама выставки", 
                          "Реклама телепередачи", 
                          "Реклама телеканала", 
                          "Реклама мероприятий для детей",
                          'Реклама места проведения мероприятия',
                          'Реклама телевизионного шоу',
                          'Реклама онлайн-кинотеатра',
                          'Реклама музыкальных концертов',
                          'Реклама партнеров мероприятия',
                          'Реклама музыкальной группы',
                          'Реклама развлекательных мероприятий',
                          'Реклама арены',
                          'Реклама концертного мероприятия',
                          'Реклама музея',
                          'Реклама музыкального фестиваля',
                          'Реклама музыкальных групп и исполнителей',
                          'Реклама платформы для просмотра сериалов',
                          'Реклама кинопремьеры',
                          'Реклама цирковых представлений',
                          'Реклама симфонических оркестров',
                          'Академия моды',
                          'VK Fest',
                          'Реклама приложения',
                          'Stand-up',
                          'Билеты на сайте',
                          'Манга',
                          'Скачай приложение',
                          'ВК Видео'
                          ],
                         [r'\bтелевизионн\w*\b', 
                          r'\bвыставк\w*\b', 
                          r'\bтелепередач\w*\b', 
                          r'\bтелеканал\w*\b', 
                          r'\bмероприят\w*\b', 
                          r'\bкинотеатр\w*\b',
                          r'\bонлайн-кинотеатр\w*\b',
                          r'\bконцерт\w*\b',
                          r'\bмузык\w*\b',
                          r'\bмузе\w*\b',
                          r'\bсериал\w*\b',
                          r'\bкинопремьер\w*\b',
                          r'\bцирк\w*\b',
                          r'\bоркестр\w*\b',
                          r'\bпесн\w*\b',
                          r'\bартист\w*\b',
                          r'\bОкко\w*\b',
                          r'\bжурнал\w*\b',
                          r'\bмюзикл\w*\b',
                          r'\bрадио\w*\b',
                          r'\bспектакл\w*\b',
                          r'\bтеатр\w*\b',
                          r'\bприложен\w*\b',
                          r'\bконтент\w*\b',
                          r'\bаниме\w*\b',
                          r'\bхудож\w*\b',
                          r'\bдирижёр\w*\b',
                          r'\bарт-холл\w*\b',
                          r'\bфортепиан\w*\b',
                          r'\bкомикс\w*\b',
                          r'\bфестивал\w*\b',
                          r'\bвидеохостинг\w*\b',
                          ]
                         ]
        qwen_results_file = "/home/tbdbj/forest_test/qwen/qwen_ds_russ_check_3_.csv"
        super().__init__(name=CTGR_NAME[0], keywords=keywords_qwen, qwen_file=qwen_results_file, new_delimiter=';')


test_ctgr = base.Category(
     name=CTGR_NAME[0], 
     detectors=[MyOcrDetector(), QwenDetector()],
     mixer=mixer.RFRScikit(MIXER_NAME)
)
#----------------------------------------------------------------------------------------------
# тест категории
#----------------------------------------------------------------------------------------------

load_diff_files()

df_res = pd.DataFrame()

def ds_russ2024y_russ2500():
    combined = ds_combine(
        ds_russ2024y,
        ds_russ2500
    )
    for row in combined:
        yield row

POS_PROMPTS = ["Реклама телевизионного сериала", 
                "Реклама выставки", 
                "Реклама телепередачи", 
                "Реклама телеканала", 
                "Реклама мероприятий для детей",
                'Реклама места проведения мероприятия',
                'Реклама телевизионного шоу',
                'Реклама онлайн-кинотеатра',
                'Реклама музыкальных концертов',
                'Реклама партнеров мероприятия',
                'Реклама музыкальной группы',
                'Реклама развлекательных мероприятий',
                'Реклама арены',
                'Реклама концертного мероприятия',
                'Реклама музея',
                'Реклама музыкального фестиваля',
                'Реклама музыкальных групп и исполнителей',
                'Реклама платформы для просмотра сериалов',
                'Реклама кинопремьеры',
                'Реклама цирковых представлений',
                'Реклама симфонических оркестров',
                ]

NEG_PROMPTS = ["документы в приложении",
               "приложение"]
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

df_res['detection_vector'] = df_res['detection_vector'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
df_res.to_csv(CSV_PATH,  index=False, encoding='utf-8', sep=',')

df = load_df_from_csv(CSV_PATH, sep=',')

metrix = print_metrix(df=df, threshold=0.4)
report(test_ctgr=test_ctgr, metrics=metrix, importance=None, dataset='ds_russ_check_1', pref='res_and_4', suf='_05_03_ds_check_1')
#----------------------------------------------------------------------------------------------
# тест обучения
#----------------------------------------------------------------------------------------------
X, Y = load_X_Y_from_csv(test_ctgr, df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

test_ctgr.mixer.fit(X_train, Y_train, property={
    'n_estimators': 100, 
    'max_depth': 50,
    'min_samples_split': 10,
    'min_samples_leaf': 5,  
    'random_state': 42    
})

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

metrix = print_metrix(Y_t=Y_test, Y_p=Y_pred, threshold=0.4)

feature_importances = feature_importances_calc(test_ctgr.mixer, X_train)

report(test_ctgr=test_ctgr, metrics=metrix, importance=feature_importances, dataset='ds_russ2024y+ds_russ2500', pref='res_and_2', suf='_05_03_ds_3_forest')
