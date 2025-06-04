# data_metrics.py
import glob
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import re
import os
import json
import pytz
from datetime import datetime
from typing import List


def load_df_from_csv(*csv_files):
    file_paths = []
    for path in csv_files:
        file_paths.extend(glob.glob(path))
    df_list = [pd.read_csv(file) for file in file_paths]
    if len(file_paths) > 1:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = df_list[0]
    df.dropna(inplace=True)
    return df


def get_feature_names(
    test_ctgr,
    clip_dim=512,
    num_pos_prompts: List[str] = None,
    num_neg_prompts: List[str] = None,
    core_phrases: List[str] = None,
    aux_phrases: List[str] = None
):
    from yolo import YOLODetector
    from ocr import OcrDetector
    from qwen import QwenDetector

    feature_names = []
    detector_ids = []

    for detector in test_ctgr.list:
        if isinstance(detector, YOLODetector):
            feature_names += [f"YOLO_{n}" for n in detector.names]
            detector_ids.append("yolo")
        elif isinstance(detector, OcrDetector):
            feature_names += [f"OCR_{n.upper()}" for n in detector.names]
            detector_ids.append("ocr")
        elif isinstance(detector, QwenDetector):
            feature_names += [f"QWEN_{n.upper()}" for n in detector.names]
            detector_ids.append("qwen")
        else:
            detector_ids.append(detector.__class__.__name__.lower())

    # CLIP
    clip_emb_names = [f'CLIP_emb_{i}' for i in range(clip_dim)]
    clip_pos_names = [f'CLIP_pos_{i}' for i in range(len(num_pos_prompts))]
    clip_neg_names = [f'CLIP_neg_{i}' for i in range(len(num_neg_prompts))]
    feature_names += clip_emb_names + clip_pos_names + clip_neg_names

    # CORE / AUX флаги
    if core_phrases:
        feature_names += [
            f'CORE_{phrase.upper().replace(" ", "_")}' for phrase in core_phrases
        ]
    if aux_phrases:
        feature_names += [
            f'AUX_{phrase.upper().replace(" ", "_")}' for phrase in aux_phrases
        ]

    # Описание пайплайна
    detectors_str = "_".join(detector_ids)
    mixer_model = test_ctgr.mixer.model
    mixer_str = type(mixer_model).__name__.lower() if mixer_model else None
    pipeline_str = f"{detectors_str} -> {mixer_str}"

    return feature_names, pipeline_str


def load_X_Y_from_csv(
    test_ctgr, 
    df,
    num_pos_prompts: List[str] = None,
    num_neg_prompts: List[str] = None,
    core_phrases: List[str] = None,
    aux_phrases: List[str] = None
):
    """
    Подготовка X, Y для обучения.
    """
    df_loaded = df.copy()
    df_loaded['detection_vector'] = df_loaded['detection_vector'].apply(ast.literal_eval)

    feature_names, _ = get_feature_names(
        test_ctgr,
        num_pos_prompts=num_pos_prompts,
        num_neg_prompts=num_neg_prompts,
        core_phrases=core_phrases,
        aux_phrases=aux_phrases
    )

    detection_df = pd.DataFrame(df_loaded['detection_vector'].tolist(), columns=feature_names)

    X = detection_df
    Y = df_loaded['category_present']

    return X, Y


def print_metrix(df=None, Y_t=None, Y_p=None, threshold=0.86):
    """
    Вывод метрик Accuracy, Precision, Recall, F1, ROC-AUC и confusion_matrix
    Возвращает:
        - словарь с метриками
        - словарь с TP, FP, TN, FN
    """
    if Y_t is not None and Y_p is not None:
        Y_test = np.array(Y_t)
        Y_pred_raw = np.array(Y_p)
    elif df is not None:
        Y_test = df['category_present'].astype(int).values
        Y_pred_raw = df['threshold_passed'].values
    else:
        raise ValueError("Переданы неверные данные")

    # Если предсказания – не бинарные, применяем порог
    if not np.array_equal(np.unique(Y_pred_raw), [0, 1]):
        Y_pred = (Y_pred_raw >= threshold).astype(int)
    else:
        Y_pred = Y_pred_raw.astype(int)

    try:
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)
        roc_auc = roc_auc_score(Y_test, Y_pred_raw)
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")

        return (
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            },
            {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
        )
    except ValueError:
        roc_auc = roc_auc_score(Y_test, Y_pred_raw)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        return (
            {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": roc_auc,
            },
            None
        )


def feature_importances_calc(mixer, X_train):
    """
    Рисует и сохраняет график важности признаков, возвращает словарь важности признаков.
    """
    feature_importances = mixer.model.feature_importances_
    feature_names = X_train.columns
    
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[sorted_idx]
    sorted_names = feature_names[sorted_idx]

    importance_dict = {name: importance for name, importance in zip(sorted_names, sorted_importances)}

    plt.figure(figsize=(12, 6))
    plt.barh(sorted_names, sorted_importances, color="skyblue")
    plt.xlabel(f"Важность признаков {mixer.name}")
    plt.title(f"График важности признаков {mixer.name}")
    plt.gca().invert_yaxis()
    os.makedirs('results_feature_importances', exist_ok=True)
    plt.savefig(f"results_feature_importances/{mixer.name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return importance_dict


def safe_filename(name, max_length=50):
    """Безопасное имя файла: если name является списком, объединяет его элементы через '_',
    затем удаляет спецсимволы и обрезает до max_length символов."""
    if isinstance(name, list):
        name = "_".join(name)
    safe = re.sub(r'[\/:*?"<>|]', '_', name)
    return safe[:max_length]


def report(test_ctgr, metrics, importance=None, dataset="ds_russ2024y", comment="Some comments", pref='results_reports', suf=''): 
    """
        Генерирует отчет с метриками модели, матрицей ошибок и важностью признаков
        Возвращает: словарь с данными отчета
    """
    tz = pytz.timezone("Europe/Moscow") 
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S.%f")
    
    metrics_dict, conf_matrix_dict = metrics
    _, pipe_str = get_feature_names(test_ctgr)
    
    report_dict = {
        "CtgrName": f"{test_ctgr.name}",  
        "DateTime": current_time, 
        "Dataset": f"{dataset}", 
        "Pipeline": pipe_str, 
        "Metrics": metrics_dict, 
        "MatrixConfusion": {
            "tp": int(conf_matrix_dict.get("tp", 0)),  # if conf_matrix_dict else 0 ???
            "tn": int(conf_matrix_dict.get("tn", 0)), 
            "fp": int(conf_matrix_dict.get("fp", 0)), 
            "fn": int(conf_matrix_dict.get("fn", 0)) 
        },
        "Importance": importance if isinstance(importance, dict) and importance else None,
        "Comments": comment
    }
    
    os.makedirs(f'{pref}', exist_ok=True)
    json_file_path = f"{pref}/{dataset}-{pipe_str.replace("->", "_")}{suf}.json"
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(report_dict, json_file, ensure_ascii=False, indent=4)
        
    return report_dict 