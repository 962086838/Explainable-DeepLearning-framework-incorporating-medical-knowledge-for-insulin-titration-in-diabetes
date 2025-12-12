from typing import List

import numpy as np
import torch
from datasets.slices.slice import slice_patient_info, slice_patient_info_targettime
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
    exam_normalize: bool = False,
    mask_last_day: bool = True,
    sugar_start_index: int = 0,
    sugar_end_index: int = 7,
    sugar_miss_maximum: int = 3,
):
    y_sugar = data_dict["last_day"]["sugar_vec_7_point"][
        :, sugar_start_index:sugar_end_index
    ]
    y_insulin = data_dict["last_day"]["insulin_vec_7_point"][
        :, sugar_start_index:sugar_end_index
    ]

    mask = np.equal(y_sugar, DEFAULT_MISSING_VALUE).sum(1) <= sugar_miss_maximum

    # mask with target sugar
    x_dict = {k: v[mask] for k, v in data_dict["feature_day"].items()}

    # if mask_last_day:
    #     x_dict = mask_feature_vector_in_last_day(x_dict)

    x_dict = {k: torch.FloatTensor(v) for k, v in x_dict.items()}

    y_sugar = torch.FloatTensor(y_sugar[mask])
    y_insulin = torch.FloatTensor(y_insulin[mask][:, :, -1])
    return x_dict, (y_sugar, y_insulin)


class MixPredictDataset(Dataset):
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
        预测固定一天7点血糖和胰岛素用的dataset
        """
        self.patient_list = patient_list
        data_padding = slice_patient_info(
            self.patient_list,
            max_feature_days=max_feature_days,
            min_feature_days=min_feature_days,
            pad_value=DEFAULT_MISSING_VALUE,
        )
        # print(data_padding['feature_day']['sugar_vec_7_point'])
        self.x, (self.y_sugar, self.y_insulin) = mask_data_with_sugar_mix(
            data_padding,
            sugar_miss_maximum=sugar_miss_maximum,
            sugar_start_index=sugar_start_index,
            sugar_end_index=sugar_end_index,
        )
        # print(self.x['sugar_vec_7_point'])

    def __len__(self):
        return self.y_sugar.shape[0]

    def __getitem__(self, index):
        x = {k: v[index] for k, v in self.x.items()}
        return x, (self.y_sugar[index], self.y_insulin[index])

class MixPredictDataset_targetpidday(Dataset):
    def __init__(
            self,
            patient_list: List,
            max_feature_days: int = 10,
            min_feature_days: int = 3,
            sugar_miss_maximum: int = 5,
            sugar_start_index: int = 0,
            sugar_end_index: int = 7,
            target_patient_day=None,
    ):
        """
        预测固定一天7点血糖和胰岛素用的dataset
        """
        self.patient_list = patient_list
        assert target_patient_day is not None
        data_padding = slice_patient_info_targettime(
            self.patient_list,
            max_feature_days=max_feature_days,
            min_feature_days=min_feature_days,
            pad_value=DEFAULT_MISSING_VALUE,
            target_patient_day=target_patient_day,
        )
        self.x, (self.y_sugar, self.y_insulin) = mask_data_with_sugar_mix(
            data_padding,
            sugar_miss_maximum=7,
            sugar_start_index=sugar_start_index,
            sugar_end_index=sugar_end_index,
        )
        # print(self.x['sugar_vec_7_point'])

    def __len__(self):
        return self.y_sugar.shape[0]

    def __getitem__(self, index):
        x = {k: v[index] for k, v in self.x.items()}
        return x, (self.y_sugar[index], self.y_insulin[index])

