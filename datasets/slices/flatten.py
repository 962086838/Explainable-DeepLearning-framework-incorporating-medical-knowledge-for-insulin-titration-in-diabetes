from collections import defaultdict
from typing import List

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm

from datasets.patient import PatientInfo
from utils.constants import DEFAULT_MISSING_VALUE


def slice_patient_info(
    patient_list: List[PatientInfo],
    max_feature_days: int = 4,
    min_feature_days: int = 2,
    pad_value: float = DEFAULT_MISSING_VALUE,
    encode_points: int = 7,
    predict_points: int = 7,
):
    """
    根据point和predict_point，对patient_list中的数据 按时间维度铺平后做切片。

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
    encode_points : int, optional
        编码的点数，比如一天7点、24点, by default 7
    predict_points : int, optional
        需要预测接下来几个点位的信息, by default 7

    Returns
    -------
    _type_
        _description_
    """
    feature_point = defaultdict(list)
    last_point = defaultdict(list)
    target_point = defaultdict(list)

    for patient in tqdm(patient_list, desc="slice_patient_info"):
        # 数据量太少跳过
        if patient.max_days - min_feature_days <= 0:
            continue

        result = slice_patient_info_single(
            patient,
            max_feature_days,
            min_feature_days,
            pad_value,
            encode_points,
            predict_points,
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


def slice_patient_info_single(
    patient: PatientInfo,
    max_feature_days: int = 4,
    min_feature_days: int = 2,
    pad_value: float = DEFAULT_MISSING_VALUE,
    point: int = 7,
    predict_point: int = 7,
):
    feature_point = defaultdict(list)
    last_point = defaultdict(list)
    target_point = defaultdict(list)

    total_point = patient.max_days * point
    max_point = max_feature_days * point
    min_point = min_feature_days * point

    vec_data = dict()

    def _inject_vec(name: str, last_dim: int):
        vec_data[name] = getattr(patient, name).reshape(total_point, last_dim)

    _inject_vec(f"day_vec_{point}_point", 1)
    _inject_vec(f"point_vec_{point}_point", 1)
    _inject_vec(f"sugar_vec_{point}_point", 1)
    _inject_vec(f"drug_vec_{point}_point", 28)
    _inject_vec(f"insulin_vec_{point}_point", 9)
    _inject_vec(f"insulin_vec_{point}_point_temp", 9)

    for pt in range(total_point + 1):
        if pt < min_point:
            continue
        _start = max(0, pt - max_point)
        _end = pt

        feature_point["patient_id"].append(patient.patient_id)
        feature_point["examination_vec"].append(patient.examination_vec)

        for k, v in vec_data.items():
            feature_point[k].append(v[_start:_end])

        last_point[f"sugar_vec_{point}_point"].append(
            vec_data[f"sugar_vec_{point}_point"][_end - predict_point : _end][:, 0]
        )
        last_point[f"insulin_vec_{point}_point"].append(
            vec_data[f"insulin_vec_{point}_point"][_end - predict_point : _end]
        )

        # target_point["sugar_vec_7_point"].append(
        #     sugar_vec_7_point[_end : _end + predict_point][:, 0]
        # )
        # target_point["insulin_vec_7_point"].append(
        #     insulin_vec_7_point[_end : _end + predict_point]
        # )

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
