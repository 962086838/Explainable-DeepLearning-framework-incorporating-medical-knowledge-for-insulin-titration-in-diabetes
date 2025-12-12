from typing import List

import numpy as np
import torch
from datasets.slices.slice import slice_patient_info
from torch.utils.data import Dataset
from utils.constants import DEFAULT_MISSING_VALUE


def mask_feature_vector_in_last_day(
    data_dict: dict, mask_value: float = DEFAULT_MISSING_VALUE
) -> dict:
    data_dict["sugar_vec_7_point"][:, -1] = mask_value
    data_dict["sugar_vec_24_point"][:, -1] = mask_value
    return data_dict


def mask_data_with_sugar(
    data_dict: dict,
    exam_normalize: bool = False,
    mask_last_day: bool = True,
    sugar_start_index: int = 0,
    sugar_end_index: int = 7,
    sugar_miss_maximum: int = 3,
):
    """
    根据last_day中的7点血糖信息，选取last_day中7点血糖缺失值不超过sugar_miss_maximum次的数据
    """
    y_sugar = data_dict["last_day"]["sugar_vec_7_point"][
        :, sugar_start_index:sugar_end_index
    ]
    mask = np.equal(y_sugar, DEFAULT_MISSING_VALUE).sum(1) <= sugar_miss_maximum

    # mask with target sugar
    x_dict = {k: v[mask] for k, v in data_dict["feature_day"].items()}

    if mask_last_day:
        x_dict = mask_feature_vector_in_last_day(x_dict)

    for k, v in x_dict.items():
        try:
            x_dict[k] = torch.FloatTensor(v)
        except Exception as e:
            print(k)
            raise e
    y_sugar = torch.FloatTensor(y_sugar[mask])
    return (x_dict, y_sugar)


class SugarPredictDataset(Dataset):
    def __init__(
        self,
        patient_list: List,
        max_feature_days: int = 10,
        min_feature_days: int = 3,
        sugar_miss_maximum: int = 5,
        sugar_start_index: int = 0,
        sugar_end_index: int = 7,
    ):
        """
        预测固定一天7点血糖用的dataset
        此为最基础的dataset，后续有mix，为血糖+胰岛素同时预测的dataset
        """
        self.patient_list = patient_list
        data_padding = slice_patient_info(
            self.patient_list,
            max_feature_days=max_feature_days,
            min_feature_days=min_feature_days,
            pad_value=DEFAULT_MISSING_VALUE,
        )

        self.x, self.y = mask_data_with_sugar(
            data_padding,
            sugar_miss_maximum=sugar_miss_maximum,
            sugar_start_index=sugar_start_index,
            sugar_end_index=sugar_end_index,
        )

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x = {k: v[index] for k, v in self.x.items()}
        y = self.y[index]
        return x, y
