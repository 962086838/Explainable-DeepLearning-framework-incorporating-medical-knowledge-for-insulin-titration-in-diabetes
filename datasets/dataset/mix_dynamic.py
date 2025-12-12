from typing import List

import numpy as np
import torch
from datasets.slices.flatten import slice_patient_info
from torch.utils.data import Dataset
from utils.constants import DEFAULT_MISSING_VALUE


def mask_feature_vector_in_last_day(
    data_dict: dict, mask_value: float = DEFAULT_MISSING_VALUE
) -> dict:
    if "sugar_vec_7_point" in data_dict:
        data_dict["sugar_vec_7_point"][:, -7:] = mask_value
    if "sugar_vec_24_point" in data_dict:
        data_dict["sugar_vec_24_point"][:, -7:] = mask_value
    return data_dict


def mask_data_with_sugar_mix(
    data_dict: dict,
    # exam_normalize: bool = False,
    # mask_last_day: bool = False,
    point: int = 7,
    sugar_miss_maximum: int = 5,
):
    y_sugar = data_dict["last_point"][f"sugar_vec_{point}_point"]
    y_insulin = data_dict["last_point"][f"insulin_vec_{point}_point"]

    mask = np.equal(y_sugar, DEFAULT_MISSING_VALUE).sum(1) <= sugar_miss_maximum
    # mask with target sugar
    x_dict = {k: v[mask] for k, v in data_dict["feature_point"].items()}

    # if mask_last_day:
    #     x_dict = mask_feature_vector_in_last_day(x_dict)

    x_dict = {k: torch.FloatTensor(v) for k, v in x_dict.items()}

    y_sugar = torch.FloatTensor(y_sugar[mask])
    y_insulin = torch.FloatTensor(y_insulin[mask][:, :, -1])

    return x_dict, (y_sugar, y_insulin)


class MixDynamicPredictDataset(Dataset):
    def __init__(
        self,
        patient_list: List,
        max_feature_days: int = 10,
        min_feature_days: int = 3,
        sugar_miss_maximum: int = 5,
        encode_points: int = 7,
        predict_points: int = 7,
    ):
        """
        融合血糖和胰岛素预测的数据，Dynamic是指不受一天7点的限定，可以任意时间段的7点,7点也是不固定的，可以是24点
        Parameters
        ----------
        patient_list : List
            患者列表
        max_feature_days : int, optional
            最长使用的天数, by default 10
        min_feature_days : int, optional
            最短使用的天数, by default 3
        sugar_miss_maximum : int, optional
            mask时最大缺失血糖数量, by default 5
        encode_points : int, optional
            编码使用的点数，7就是7点血糖, by default 7
        predict_points : int, optional
            预测后续多少点的血糖, by default 7
        """
        self.patient_list = patient_list
        data_padding = slice_patient_info(
            self.patient_list,
            max_feature_days=max_feature_days,
            min_feature_days=min_feature_days,
            pad_value=DEFAULT_MISSING_VALUE,
            encode_points=encode_points,
            predict_points=predict_points,
        )

        self.x, (self.y_sugar, self.y_insulin) = mask_data_with_sugar_mix(
            data_padding,
            sugar_miss_maximum=sugar_miss_maximum,
            point=encode_points,
        )

    def __len__(self):
        return self.y_sugar.shape[0]

    def __getitem__(self, index):
        x = {k: v[index] for k, v in self.x.items()}
        return x, (self.y_sugar[index], self.y_insulin[index])
