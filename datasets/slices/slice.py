from collections import defaultdict
from typing import List

import numpy as np
from keras_preprocessing.sequence import pad_sequences

from datasets.patient import PatientInfo
from utils.constants import DEFAULT_MISSING_VALUE


def slice_patient_info(
    patient_list: List[PatientInfo],
    max_feature_days: int = 4,
    min_feature_days: int = 2,
    pad_value: float = DEFAULT_MISSING_VALUE,
):
    """
    固定对day层次的信息做切片。

    Parameters
    ----------
    patient_list : List[PatientInfo]
        _description_
    max_feature_days : int, optional
        _description_, by default 4
    min_feature_days : int, optional
        _description_, by default 2
    pad_value : float, optional
        _description_, by default DEFAULT_MISSING_VALUE

    Returns
    -------
    _type_
        _description_
    """
    feature_day = defaultdict(list)
    last_day = defaultdict(list)
    target_day = defaultdict(list)

    for patient in patient_list:
        # 数据量太少跳过
        if patient.max_days - min_feature_days <= 0:
            continue

        result = slice_patient_info_single(
            patient, max_feature_days, min_feature_days, pad_value
        )

        for k, v in result["feature_day"].items():
            feature_day[k].append(v)
        for k, v in result["last_day"].items():
            last_day[k].append(v)
        for k, v in result["target_day"].items():
            target_day[k].append(v)
    for k, v in feature_day.items():
        feature_day[k] = np.concatenate(v)
    for k, v in last_day.items():
        last_day[k] = np.concatenate(v)
    for k, v in target_day.items():
        target_day[k] = np.concatenate(v)
    return {"feature_day": feature_day, "last_day": last_day, "target_day": target_day}


def slice_patient_info_single(
    patient: PatientInfo,
    max_feature_days: int = 4,
    min_feature_days: int = 2,
    pad_value: float = DEFAULT_MISSING_VALUE,
):
    feature_day = defaultdict(list)
    last_day = defaultdict(list)
    target_day = defaultdict(list)

    feature_items = [
        "day_vec_7_point",
        "day_vec_24_point",
        "point_vec_7_point",
        "point_vec_24_point",
        "sugar_vec_7_point",
        "sugar_vec_24_point",
        "drug_vec_1_point",
        "drug_vec_7_point",
        "drug_vec_24_point",
        "insulin_vec_4_point",
        "insulin_vec_7_point",
        "insulin_vec_24_point",
        "insulin_vec_7_point_temp",
        "insulin_vec_24_point_temp",
    ]
    for day in range(patient.max_days):
        if day < min_feature_days:
            continue
        _start = max(0, day - max_feature_days)
        _end = day

        feature_day["patient_id"].append(patient.patient_id)
        feature_day["patient_day"].append(day)
        feature_day["examination_vec"].append(patient.examination_vec)

        for item in feature_items:
            vec = getattr(patient, item)
            feature_day[item].append(vec[_start:_end])
            last_day[item].append(vec[day - 1])

            if day == patient.max_days:
                vec_ = np.zeros_like(vec[day - 1])
                vec_.fill(DEFAULT_MISSING_VALUE)
                target_day[item].append(vec_)
            else:
                target_day[item].append(vec[day])

    for item in feature_items:
        feature_day[item] = pad_sequences(
            feature_day[item], maxlen=max_feature_days, dtype="float32", value=pad_value
        )

    return {"feature_day": feature_day, "last_day": last_day, "target_day": target_day}


def slice_patient_info_targettime(
        patient_list: List[PatientInfo],
        max_feature_days: int = 4,
        min_feature_days: int = 2,
        pad_value: float = DEFAULT_MISSING_VALUE,
        target_patient_day=None,
):
    """
    固定对day层次的信息做切片。

    Parameters
    ----------
    patient_list : List[PatientInfo]
        _description_
    max_feature_days : int, optional
        _description_, by default 4
    min_feature_days : int, optional
        _description_, by default 2
    pad_value : float, optional
        _description_, by default DEFAULT_MISSING_VALUE

    Returns
    -------
    _type_
        _description_
    """
    feature_day = defaultdict(list)
    last_day = defaultdict(list)
    target_day = defaultdict(list)

    for patient, each_target_day in zip(patient_list, target_patient_day):
        # 数据量太少跳过
        if patient.max_days - min_feature_days <= 0:
            continue

        result = slice_patient_info_single_targettime(
            patient, max_feature_days, min_feature_days, pad_value, target_patient_day=each_target_day,
        )

        for k, v in result["feature_day"].items():
            feature_day[k].append(v)
        for k, v in result["last_day"].items():
            last_day[k].append(v)
        for k, v in result["target_day"].items():
            target_day[k].append(v)
    for k, v in feature_day.items():
        feature_day[k] = np.concatenate(v)
    for k, v in last_day.items():
        last_day[k] = np.concatenate(v)
    for k, v in target_day.items():
        target_day[k] = np.concatenate(v)
    return {"feature_day": feature_day, "last_day": last_day, "target_day": target_day}

def slice_patient_info_single_targettime(
    patient: PatientInfo,
    max_feature_days: int = 4,
    min_feature_days: int = 2,
    pad_value: float = DEFAULT_MISSING_VALUE,
    target_patient_day=None,
):
    feature_day = defaultdict(list)
    last_day = defaultdict(list)
    target_day = defaultdict(list)

    feature_items = [
        "day_vec_7_point",
        "day_vec_24_point",
        "point_vec_7_point",
        "point_vec_24_point",
        "sugar_vec_7_point",
        "sugar_vec_24_point",
        "drug_vec_1_point",
        "drug_vec_7_point",
        "drug_vec_24_point",
        "insulin_vec_4_point",
        "insulin_vec_7_point",
        "insulin_vec_24_point",
        "insulin_vec_7_point_temp",
        "insulin_vec_24_point_temp",
    ]

    day = target_patient_day
    _start = max(0, day - max_feature_days)
    _end = day

    feature_day["patient_id"].append(patient.patient_id)
    feature_day["examination_vec"].append(patient.examination_vec)

    for item in feature_items:
        vec = getattr(patient, item)
        feature_day[item].append(vec[_start:_end])
        last_day[item].append(vec[day - 1])

        if day == patient.max_days:
            vec_ = np.zeros_like(vec[day - 1])
            vec_.fill(DEFAULT_MISSING_VALUE)
            target_day[item].append(vec_)
        else:
            target_day[item].append(vec[day])

    for item in feature_items:
        if 'drug' in item:
            feature_day[item] = pad_sequences(
                feature_day[item], maxlen=max_feature_days, dtype="float32", value=0
            )
        else:
            feature_day[item] = pad_sequences(
                feature_day[item], maxlen=max_feature_days, dtype="float32", value=pad_value
            )

    return {"feature_day": feature_day, "last_day": last_day, "target_day": target_day}

