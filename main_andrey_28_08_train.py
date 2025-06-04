from __future__ import annotations
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from categories.category import Category
from detectors.yolo import YOLODetector
from detectors.ocr import OCRDetector
from detectors.qwen import QwenDetector
from mixers.rfr_scikit import RFRScikit
from utils import (
    Dataset,
    DATASET_PATHS,
    combine,
    ds_check_ctgr,
    load_diff_files,
    load_X_Y_from_csv,
    print_metrix,
    feature_importances_calc,
    safe_filename,
    calc_local_memory,
)

CTGR_NAME = "28.08 Застройщик"
MIXER_NAME = "mix_28_08"


def dataset_source():
    ds1 = Dataset("russ9200", DATASET_PATHS["russ9200"])
    ds2 = Dataset("russ2500", DATASET_PATHS["russ2500"])
    for row in combine(ds1, ds2):
        yield row


def build_category() -> Category:
    yolo_det = YOLODetector(
        name=CTGR_NAME,
        classes=["developer_logo"],
        detectors=["yolo_developer_logo"],
    )
    ocr_det = OCRDetector(
        name=CTGR_NAME,
        texts=["застройщик", "специализированный застройщик", "застройщик ООО"],
    )
    qwen_det = QwenDetector(
        name=CTGR_NAME,
        keywords=[["Застройщик"], [r"\\bзастройщ\\w*\\b"]],
        qwen_file="old_files/qwen/qwen_ds_russ_check_3_.csv",
        delimiter=";",
    )
    return Category(
        name=CTGR_NAME,
        detectors=[yolo_det, ocr_det, qwen_det],
        mixers={"rfr": RFRScikit},
    )


def main():
    load_diff_files()
    category = build_category()
    rows = []
    for file_name, y, path in ds_check_ctgr([CTGR_NAME], dataset_source):
        local = calc_local_memory(path)
        if local is None:
            continue
        vec = category.calc_vec(local)
        rows.append({
            "file_name": file_name,
            "current_file_name": os.path.basename(path),
            "category_present": y,
            "detection_vector": vec.tolist(),
        })
    df = pd.DataFrame(rows)
    csv_path = f"train_{safe_filename(CTGR_NAME)}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    X, Y = load_X_Y_from_csv(category, df)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y
    )
    mixer = RFRScikit(MIXER_NAME)
    mixer.fit(X_train.values, Y_train.values, properties={"n_estimators": 100})
    mixer.save(MIXER_NAME)
    Y_pred = mixer.predict(X_test.values)
    print_metrix(Y_t=Y_test, Y_p=Y_pred)
    feature_importances_calc(mixer, X_train)


if __name__ == "__main__":
    main()
