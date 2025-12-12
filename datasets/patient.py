import os
import random
from typing import List

import numpy as np
import pendulum
from datasets.utils import DatasetEncodeUtil, read_json
from utils.constants import PATIENT_BASE_DIR


def parse_datetime(dts: str):
    try:
        return pendulum.parse(dts)
    except Exception as e:
        return None


def load_patient_list(
    base_dir: str = PATIENT_BASE_DIR,
    patient_filter_path: str = "./resources/train_filename_list.txt",
    limit: int = 100,
    shuffle: bool = True,
) -> List["PatientInfo"]:
    PatientInfo.base_dir = base_dir
    PatientInfo.util = DatasetEncodeUtil()
    file_list = os.listdir(base_dir)
    patient_list = []
    err_count = 0
    with open(patient_filter_path) as fin:
        _content = fin.read().split("\n")
        filter_filename_list = sorted(list(set(_content)))[:limit]
        print('len(filter_filename_list)', len(filter_filename_list))
    for filename in sorted(file_list):
    # for filename in file_list:
        if filename not in filter_filename_list:
            continue
        try:
            patient_list.append(PatientInfo(filename))
        except Exception as e:
            err_count += 1
            raise e
            continue
    # patient_list = [PatientInfo(filename) for filename in file_list]
    if shuffle:
        random.shuffle(patient_list)
    for each in patient_list:
        print(each.patient_id)
    # print("patient_list: ", patient_list)
    print("err_count: ", err_count)
    return patient_list


class PatientInfo:
    base_dir: str
    util: DatasetEncodeUtil

    def __init__(self, filename: str = None, patient_data: dict = None):
        assert filename or patient_data, "filename 和 patient_data 都为空"
        if filename:
            self.filename = filename
            self.patient_data = read_json(os.path.join(self.base_dir, filename))
        else:
            self.patient_data = patient_data
        self.patient_id = int(self.patient_data["patient_id"])

        self.max_days = 0

        self.has_temp = False  # 是否有临时事件
        self.has_temp_insulin = False  # 是否有临时胰岛素
        self.has_hypoglycemia = False  # 是否存在过低血糖

        self.temp_list = []
        self.temp_insulin_time_list = []
        self.temp_count = 0
        self.temp_insulin_count = 0

        self.day_vec_7_point: np.ndarray = None  # (day, 7)
        self.day_vec_24_point: np.ndarray = None  # (day, 24)

        # 时刻点
        self.point_vec_7_point: np.ndarray = None  # (day, 7)
        self.point_vec_24_point: np.ndarray = None  # (day, 24)

        self.examination_vec: np.ndarray = None  # (78, )

        self.sugar_vec_7_point: np.ndarray = None  # (day, 7)
        self.sugar_vec_24_point: np.ndarray = None  # (day, 24)

        self.drug_vec_1_point: np.ndarray = None  # (day, 28)
        self.drug_vec_7_point: np.ndarray = None  # (day, 7, 28)
        self.drug_vec_24_point: np.ndarray = None  # (day, 24, 28)

        self.insulin_vec_4_point: np.ndarray = None  # (day, 4, *)
        self.insulin_vec_7_point: np.ndarray = None  # (day, 7, *)
        self.insulin_vec_24_point = None  # (day, 24, *)

        self.insulin_vec_7_point_temp: np.ndarray = None  # (day, 7, *)
        self.insulin_vec_24_point_temp: np.ndarray = None  # (day, 24, *)

        self.preprocess_patient_data()


    def preprocess_patient_data(self):
        self.examination_vec = self.util.basic_info_to_examination_vec(
            self.patient_data["basic_info"]
        )
        all_sugar_vec_7_point = []
        all_sugar_vec_24_point = []

        all_drug_vec_1_point = []
        all_drug_vec_7_point = []
        all_drug_vec_24_point = []

        all_insulin_vec_4_point = []
        all_insulin_vec_7_point = []
        all_insulin_vec_24_point = []

        all_insulin_vec_7_point_temp = []
        all_insulin_vec_24_point_temp = []

        for day, timeline in self.patient_data["days"].items():
            self.max_days = max(self.max_days, int(day) + 1)

            seven_points = timeline["daily_routine"]
            temporary_events = timeline["temporary_events"]

            if temporary_events:
                self.has_temp = True
                self.temp_list.append(temporary_events)
                self.temp_count += len(temporary_events)
                for temp in temporary_events:
                    if temp["action"] == "injecting_insulin":
                        self.has_temp_insulin = True
                        self.temp_insulin_count += 1
                        self.temp_insulin_time_list.append(temp["time"])

            sugar_vec_7_point = self.util.sugar_7_point_encoder(seven_points)
            sugar_vec_24_point = self.util.sugar_24_point_encoder(
                seven_points, temporary_events
            )

            drug_vec_1_point = self.util.drug_1_point_encoder(seven_points)
            drug_vec_7_point = self.util.drug_7_point_encoder(seven_points)
            drug_vec_24_point = self.util.drug_24_point_encoder(seven_points)

            insulin_vec_4_point = self.util.insulin_4_point_encoder(seven_points)
            insulin_vec_7_point = self.util.insulin_7_point_encoder(seven_points)
            insulin_vec_24_point = self.util.insulin_24_point_encoder(
                seven_points, temporary_events
            )

            temp_insulin_7_point_vec = self.util.temporary_insulin_7_point_encoder(
                temporary_events
            )
            temp_insulin_24_point_vec = self.util.temporary_insulin_24_point_encoder(
                temporary_events
            )

            all_sugar_vec_7_point.append(sugar_vec_7_point)
            all_sugar_vec_24_point.append(sugar_vec_24_point)

            all_drug_vec_1_point.append(drug_vec_1_point)
            all_drug_vec_7_point.append(drug_vec_7_point)
            all_drug_vec_24_point.append(drug_vec_24_point)

            all_insulin_vec_4_point.append(insulin_vec_4_point)
            all_insulin_vec_7_point.append(insulin_vec_7_point)
            all_insulin_vec_24_point.append(insulin_vec_24_point)

            all_insulin_vec_7_point_temp.append(temp_insulin_7_point_vec)
            all_insulin_vec_24_point_temp.append(temp_insulin_24_point_vec)

        self.sugar_vec_7_point = np.array(all_sugar_vec_7_point)
        self.sugar_vec_24_point = np.array(all_sugar_vec_24_point)

        self.drug_vec_1_point = np.array(all_drug_vec_1_point)
        self.drug_vec_7_point = np.array(all_drug_vec_7_point)
        self.drug_vec_24_point = np.array(all_drug_vec_24_point)

        self.insulin_vec_4_point = np.array(all_insulin_vec_4_point)
        self.insulin_vec_7_point = np.array(all_insulin_vec_7_point)
        self.insulin_vec_24_point = np.array(all_insulin_vec_24_point)

        self.insulin_vec_7_point_temp = np.array(all_insulin_vec_7_point_temp)
        self.insulin_vec_24_point_temp = np.array(all_insulin_vec_24_point_temp)

        self.day_vec_7_point = np.arange(self.max_days).reshape(-1, 1).repeat(7, axis=1)
        self.day_vec_24_point = (
            np.arange(self.max_days).reshape(-1, 1).repeat(24, axis=1)
        )

        self.point_vec_7_point = np.repeat(
            np.arange(7).reshape(1, -1), self.max_days, axis=0
        )
        self.point_vec_24_point = np.repeat(
            np.arange(24).reshape(1, -1), self.max_days, axis=0
        )

    def show_shape(self):

        print("day_vec_7_point", self.day_vec_7_point.shape)
        print("day_vec_24_point", self.day_vec_24_point.shape)
        print("point_vec_7_point", self.point_vec_7_point.shape)
        print("point_vec_24_point", self.point_vec_24_point.shape)

        print("examination_vec", self.examination_vec.shape)

        print("sugar_vec_7_point", self.sugar_vec_7_point.shape)
        print("sugar_vec_24_point", self.sugar_vec_24_point.shape)

        print("drug_vec_1_point", self.drug_vec_1_point.shape)
        print("drug_vec_7_point", self.drug_vec_7_point.shape)
        print("drug_vec_24_point", self.drug_vec_24_point.shape)

        print("insulin_vec_4_point", self.insulin_vec_4_point.shape)
        print("insulin_vec_7_point", self.insulin_vec_7_point.shape)
        print("insulin_vec_24_point", self.insulin_vec_24_point.shape)

        print("insulin_vec_7_point_temp", self.insulin_vec_7_point_temp.shape)
        print("insulin_vec_24_point_temp", self.insulin_vec_24_point_temp.shape)
