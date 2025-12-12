from collections import defaultdict
from typing import List

import numpy as np
from keras_preprocessing.sequence import pad_sequences

from datasets.patient import PatientInfo
from utils.constants import DEFAULT_MISSING_VALUE


def point_slice_patient_info(
    patient_list: List[PatientInfo],
    max_feature_days: int = 4,
    min_feature_days: int = 2,
    pad_value: float = DEFAULT_MISSING_VALUE,
):
    feature_point = defaultdict(list)
    last_point = defaultdict(list)
    target_point = defaultdict(list)

    for patient in patient_list:
        # 数据量太少跳过
        if patient.max_days - min_feature_days <= 0:
            continue

        result = point_slice_patient_info_single(
            patient, max_feature_days, min_feature_days, pad_value
        )

        for k, v in result["feature_point"].items():
            feature_point[k].append(v)
        for k, v in result["last_point"].items():
            last_point[k].append(v)
        for k, v in result["target_point"].items():
            target_point[k].append(v)
    for k, v in feature_point.items():
        feature_point[k] = np.concatenate(v)
    for k, v in last_point.items():
        last_point[k] = np.concatenate(v)
    for k, v in target_point.items():
        target_point[k] = np.concatenate(v)
    return {
        "feature_point": feature_point,
        "last_point": last_point,
        "target_point": target_point,
    }


def point_slice_patient_info_single(
    patient: PatientInfo,
    max_feature_days: int = 4,
    min_feature_days: int = 2,
    pad_value: float = DEFAULT_MISSING_VALUE,
    point: int = 7,
):
    feature_point = defaultdict(list)
    last_point = defaultdict(list)
    target_point = defaultdict(list)

    total_point = patient.max_days * point
    max_point = max_feature_days * point
    min_point = min_feature_days * point

    day_vec_7_point = patient.day_vec_7_point.reshape(total_point, 1)
    point_vec_7_point = (
        np.arange(7)
        .reshape(1, 7)
        .repeat(patient.max_days, axis=0)
        .reshape(total_point, 1)
    )
    sugar_vec_7_point = patient.sugar_vec_7_point.reshape(total_point, 1)
    drug_vec_7_point = patient.drug_vec_7_point.reshape(total_point, 28)
    insulin_vec_7_point = patient.insulin_vec_7_point.reshape(total_point, 9)
    insulin_vec_7_point_temp = patient.insulin_vec_7_point_temp.reshape(total_point, 9)

    for point in range(total_point):
        if point < min_point:
            continue
        _start = max(0, point - max_point)
        _end = point

        feature_point["patient_id"].append(patient.patient_id)
        feature_point["examination_vec"].append(patient.examination_vec)

        feature_point["day_vec_7_point"].append(day_vec_7_point[_start:_end])
        feature_point["point_vec_7_point"].append(point_vec_7_point[_start:_end])
        feature_point["sugar_vec_7_point"].append(sugar_vec_7_point[_start:_end])
        feature_point["drug_vec_7_point"].append(drug_vec_7_point[_start:_end])
        feature_point["insulin_vec_7_point"].append(insulin_vec_7_point[_start:_end])
        feature_point["insulin_vec_7_point_temp"].append(
            insulin_vec_7_point_temp[_start:_end]
        )

        last_point["sugar_vec_7_point"].append(sugar_vec_7_point[_end])

    for k, v in feature_point.items():
        if k not in ["patient_id", "examination_vec"]:
            feature_point[k] = pad_sequences(
                v, maxlen=max_point, dtype="float32", value=pad_value
            )
    return {
        "feature_point": feature_point,
        "last_point": last_point,
        "target_point": target_point,
    }
