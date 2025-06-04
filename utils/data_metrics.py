from __future__ import annotations
import ast
import glob
import os
import re
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt


def load_df_from_csv(*csv_files: str):
    paths = []
    for p in csv_files:
        paths.extend(glob.glob(p))
    df_list = [pd.read_csv(p) for p in paths]
    df = pd.concat(df_list, ignore_index=True) if len(df_list) > 1 else df_list[0]
    df.dropna(inplace=True)
    return df


def _get_feature_names(test_ctgr, clip_dim=512):
    names = []
    for det in test_ctgr.detectors:
        names.extend(det.features)
    clip_names = [f"clip_{i}" for i in range(clip_dim)]
    names.extend(clip_names)
    pipe = "detectors->mixer"
    return names, pipe


def load_X_Y_from_csv(test_ctgr, df):
    df_loaded = df.copy()
    df_loaded['detection_vector'] = df_loaded['detection_vector'].apply(ast.literal_eval)
    feature_names, _ = _get_feature_names(test_ctgr)
    detection_df = pd.DataFrame(df_loaded['detection_vector'].tolist(), columns=feature_names)
    X = detection_df
    Y = df_loaded['category_present']
    return X, Y


def print_metrix(df=None, Y_t=None, Y_p=None, threshold=0.5):
    if Y_t is None or Y_p is None:
        Y_test = df['category_present'].values
        Y_pred_raw = df['predict_forest'].values
    else:
        Y_test = np.array(Y_t)
        Y_pred_raw = np.array(Y_p)
    if not np.array_equal(np.unique(Y_pred_raw), [0, 1]):
        Y_pred = (Y_pred_raw >= threshold).astype(int)
    else:
        Y_pred = Y_pred_raw.astype(int)
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
    return ({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}, {"tp": tp, "fp": fp, "tn": tn, "fn": fn})


def feature_importances_calc(mixer, X_train):
    if mixer.model is None:
        return None
    importances = mixer.model.feature_importances_
    names = X_train.columns
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = names[sorted_idx]
    sorted_imp = importances[sorted_idx]
    plt.figure(figsize=(12, 6))
    plt.barh(sorted_names, sorted_imp, color="skyblue")
    plt.gca().invert_yaxis()
    os.makedirs('results_feature_importances', exist_ok=True)
    plt.savefig(f'results_feature_importances/{mixer.name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return {n: float(i) for n, i in zip(sorted_names, sorted_imp)}


def safe_filename(name, max_length=50):
    if isinstance(name, list):
        name = "_".join(name)
    safe = re.sub(r'[\\/:*?"<>|]', '_', name)
    return safe[:max_length]
