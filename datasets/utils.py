from collections import defaultdict
import json
import os
import re
from typing import Dict, List

import numpy as np

from utils.constants import DEFAULT_MISSING_VALUE

TIME_INTERVAL = [6, 8.5, 10.5, 13, 16.5, 19, 21]


def read_json(filename: str) -> dict:
    with open(filename, "r", encoding='utf-8') as fin:
        return json.load(fin)


def clean_insulin_name(name: str) -> str:
    # 去掉一些特殊标注，不然种类太多
    return re.sub(r"\(危A\)|\[基\]|\(警A\)|\(甲0\%\)", "", name)


def point_24_to_7(point: float) -> int:
    for i in range(len(TIME_INTERVAL) - 1):
        if TIME_INTERVAL[i] <= point < TIME_INTERVAL[i + 1]:
            return i
    if point > TIME_INTERVAL[-1]:
        return 6


class Drug:
    name_list: List[str] = None
    class_list: List[str] = None
    dc = defaultdict(int)

    def __init__(
        self,
        name: str,
        alias: str,
        dosage_floor: float,
        dosage_ceil: float,
        onset_time_floor: float,
        onset_time_ceil: float,
        half_life_floor: float,
        half_life_ceil: float,
        classes: str,
    ):
        """
        用来处理 药品 的类，实例化单个药品

        Parameters
        ----------
        name : str
            药品名称
        alias : str
            别名
        dosage_floor : float
            剂量范围-最低值
        dosage_ceil : float
            剂量范围-最高值
        onset_time_floor : float
            起效时间-最快
        onset_time_ceil : float
            起效时间-最慢
        half_life_floor : float
            半衰期-最低值
        half_life_ceil : float
            半衰期-最高值
        classes : str
            药品分类
        """
        self.name = name
        self.alias = alias
        self.dosage_floor = dosage_floor
        self.dosage_ceil = dosage_ceil
        self.onset_time_floor = onset_time_floor
        self.onset_time_ceil = onset_time_ceil
        self.half_life_floor = half_life_floor
        self.half_life_ceil = half_life_ceil
        self.classes = classes

        self.name_index = self.name_list.index(self.name)
        self.class_index = self.class_list.index(self.classes)

        self.default_onehot = self.encode_with_name(0)
        self.default_full_info_code = self.encode_with_full_info(0)

    def encode_with_name(self, value) -> List[float]:
        # 用name的方式给药品编码
        v = [DEFAULT_MISSING_VALUE] * len(self.name_list)
        v[self.name_index] = value
        return v

    def encode_with_full_info(self, value) -> List[float]:
        # 用所有信息给胰岛素编码
        v = [
            self.name_index,
            self.class_index,
            self.dosage_floor,
            self.dosage_ceil,
            self.onset_time_floor,
            self.onset_time_ceil,
            self.half_life_floor,
            self.half_life_ceil,
            value,
        ]
        return v


class Insulin:
    classification_list: List[str] = None
    name_list: List[str] = None

    def __init__(
        self,
        name: str,
        alias: str,
        classification: str,
        onset_time_floor: float,
        onset_time_ceil: float,
        peak_time_floor: float,
        peak_time_ceil: float,
        duration_floor: float,
        duration_ceil: float,
    ):
        """
        用来处理 胰岛素 的类，实例化单个胰岛素

        Parameters
        ----------
        name : str
            胰岛素名称
        alias : str
            胰岛素别名，在别名中的胰岛素会统一归类到此胰岛素
        classification : str
            用第几个分类
        onset_time_floor : float
            起效时间-最快
        onset_time_ceil : float
            起效时间-最慢
        peak_time_floor : float
            峰值时间-至少
        peak_time_ceil : float
            峰值时间-至多
        duration_floor : float
            持续时间-至少
        duration_ceil : float
            持续时间-至多
        """
        self.name = name
        self.alias = alias
        self.classification = classification
        self.onset_time_floor = onset_time_floor
        self.onset_time_ceil = onset_time_ceil
        self.peak_time_floor = peak_time_floor
        self.peak_time_ceil = peak_time_ceil
        self.duration_floor = duration_floor
        self.duration_ceil = duration_ceil

        self.name_index = self.name_list.index(self.name)
        self.classification_index = self.classification_list.index(self.classification)

    def encode_with_classification(self, value) -> List[float]:
        # 用分类的方式给胰岛素编码
        v = [DEFAULT_MISSING_VALUE] * len(self.classifications)
        v[self.classification_index] = value
        return v

    def encode_with_full_info(self, value) -> List[float]:
        # 用所有信息给胰岛素编码
        v = [
            self.name_index,
            self.classification_index,
            self.onset_time_floor,
            self.onset_time_ceil,
            self.peak_time_floor,
            self.peak_time_ceil,
            self.duration_floor,
            self.duration_ceil,
            value,
        ]
        return v


class DrugHub:
    def __init__(
        self,
        filename: str,
        encode_mode: str = "full",
    ):
        self.filename = filename
        self.encode_mode = encode_mode
        self.data = read_json(self.filename)
        self.name_list = [i["name"] for i in self.data["drug_list"]]
        self.class_list = self.data["class_list"]

        self.drug_list: List[Drug] = list()
        self.drug_mapping: Dict[str, Drug] = dict()

        self.load_drug_list(self.data["drug_list"])
        self.full_info_code: np.ndarray = self.load_default_code()

    def load_drug_list(self, data_list: List[dict]):
        Drug.name_list = self.name_list
        Drug.class_list = self.class_list

        for data in data_list:
            drug = Drug(**data)
            self.drug_list.append(drug)
            for alias in drug.alias:
                self.drug_mapping[alias] = drug

    def load_default_code(self):
        full_info_code = []
        for drug in self.drug_list:
            full_info_code.append(drug.default_full_info_code)
        full_info_code = np.array(full_info_code)
        return full_info_code

    def encode_drug(self, name: str, value: float) -> List[float]:
        drug = self.drug_mapping[name]
        if self.encode_mode == "full":
            return drug.encode_with_full_info(value)
        elif self.encode_mode == "class":
            return drug.encode_with_classification(value)

    def encode_drug_list(self, data_list: List[dict]):
        vec = self.full_info_code.copy()
        for data in data_list:
            drug = self.drug_mapping[data["drug"]]
            vec[drug.name_index][-1] = data["value"]
            Drug.dc[drug.name] += 1
        return vec

    def encode_drug_1_points(self, seven_points_drug: dict):
        drug_list = list()
        for data in seven_points_drug.values():
            drug_list.extend(data["taking_hypoglycemic_drugs"])
        drug_vec = self.encode_drug_list(drug_list)
        return drug_vec

    def encode_drug_7_points(self, seven_points_drug: dict):
        drug_vec_list = []
        for point, data in seven_points_drug.items():
            drug_vec = self.encode_drug_list(data["taking_hypoglycemic_drugs"])
            drug_vec_list.append(drug_vec)
        return drug_vec_list


class InsulinEncoder:
    len_dc = defaultdict(int)

    def __init__(
        self,
        filename: str,
        insulin_classification_id: int = 0,
        encode_mode: str = "full",
    ):
        """
        所有胰岛素的合集，用来编码胰岛素

        Parameters
        ----------
        filename : str
            胰岛素信息的 文件名称
        insulin_classification_id : int, optional
            _description_, by default 0
        encode_mode : str, optional
            编码方式, class or full, by default "full"
        """
        self.filename = filename
        self.insulin_classification_id = insulin_classification_id
        self.encode_mode = encode_mode
        self.data = read_json(self.filename)
        self.classification_list = self.data["classifications"][
            self.insulin_classification_id
        ]
        self.insulin_list: List[Insulin] = list()
        self.insulin_mapping: Dict[str, Insulin] = dict()

        self.load_insulin_list(self.data["insulin_list"])

    def load_insulin_list(self, data_list: List[dict]):
        Insulin.classification_list = self.classification_list
        Insulin.name_list = [i["name"] for i in data_list]

        for data in data_list:
            classes = data.pop("classes")
            data["classification"] = classes[self.insulin_classification_id]
            insulin = Insulin(**data)
            self.insulin_list.append(insulin)
            for alias in insulin.alias:
                self.insulin_mapping[alias] = insulin

    def encode_insulin(self, name: str, value: float) -> List[float]:
        insulin = self.insulin_mapping[name]
        if self.encode_mode == "full":
            return insulin.encode_with_full_info(value)
        elif self.encode_mode == "class":
            return insulin.encode_with_classification(value)


class ExaminationEncoder:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = read_json("./resources/required_examinations.json")
        self.examination_list: List = list(self.data["data"].keys())
        self.examination_list += ["age", "gender"]
        self.examination_len = len(self.examination_list)

    def encode_basic_info(self, basic_info: dict) -> np.ndarray:
        vec = [DEFAULT_MISSING_VALUE] * self.examination_len
        for k, v in basic_info.items():
            if k in self.examination_list:
                vec[self.examination_list.index(k)] = v
        return np.array(vec)


class WeatherEncoder:
    def __init__(self, filename: str):
        self.filename = filename
        self.weather_list = read_json(filename)
        for weather in self.weather_list:
            weather["day"] = int(weather["date"].split("-")[-1])
        self.weather_map = {i["date"]: i for i in self.weather_list}

    def encode_weather_with_date(self, date: str):
        weather = self.weather_map.get(date)
        return self.encode_weather(weather)

    def encode_weather(self, weather: dict):
        if weather:
            return [
                weather["day"],
                weather["week"],
                weather["temperature_max"],
                weather["temperature_min"],
            ]
        else:
            return [
                DEFAULT_MISSING_VALUE,
                DEFAULT_MISSING_VALUE,
                DEFAULT_MISSING_VALUE,
                DEFAULT_MISSING_VALUE,
            ]


class DrugEncoder:
    def __init__(self, filename: str):
        self.filename = filename
        self.drug_list: List = list(read_json("./resources/needed_drugs.json"))
        self.drug_len: int = len(self.drug_list)
        self.drug_hub: DrugHub = DrugHub("./resources/drug_detail.json")

    def encode_drug(self, drug_list: List[dict]):
        drug_vec = [0] * self.drug_len
        for drug in drug_list:
            drug_vec[self.drug_list.index(drug["drug"])] += drug["value"]
            # self.drug_hub.encode_drug(drug["drug"], drug["value"])
        return drug_vec

    def encode_drug_1_points(self, seven_points_drug: dict):
        drug_list = list()
        for data in seven_points_drug.values():
            drug_list.extend(data["taking_hypoglycemic_drugs"])
        drug_vec = self.encode_drug(drug_list)
        return drug_vec

    def encode_drug_7_points(self, seven_points_drug: dict):
        drug_vec_list = []
        for point, data in seven_points_drug.items():
            drug_vec = self.encode_drug(data["taking_hypoglycemic_drugs"])
            drug_vec_list.append(drug_vec)
        return drug_vec_list


class DatasetEncodeUtil:
    insulin_len_dc = defaultdict(int)
    # 对数据做编码等
    def __init__(
        self,
        resource_dir: str = "./resources",
        insulin_classification_id: int = 0,
        insulin_encode_mode: str = "full",
    ):
        self.resource_dir = resource_dir
        self.insulin_classification_id = insulin_classification_id

        self.drug_names: List = list(read_json("./resources/needed_drugs.json"))

        self.insulin_to_type = read_json("./resources/insulin_to_type.json")
        self.insulin_classification = self.insulin_to_type["classifications"][0]
        self.insulin_dim = len(self.insulin_classification)
        self.insulin_to_type_id = dict()
        for key, value in self.insulin_to_type["insulin_to_type"].items():
            self.insulin_to_type_id[key] = self.insulin_classification.index(
                value[self.insulin_classification_id]
            )
        self.insulin_encoder = InsulinEncoder(
            os.path.join(resource_dir, "insulin_detail.json"),
            insulin_classification_id,
            insulin_encode_mode,
        )
        self.examination_encoder = ExaminationEncoder(
            os.path.join(resource_dir, "required_examinations.json"),
        )
        self.drug_encoder = DrugEncoder(os.path.join(resource_dir, "needed_drugs.json"))
        # self.drug_encoder = DrugHub(os.path.join(resource_dir, "drug_detail.json"))

        self.weather_encoder = WeatherEncoder("./resources/weather.json")
        self.patient_ex_info = {
            i["file_name"]: i for i in read_json("./resources/patient_ex_info.json")
        }

    def basic_info_to_examination_vec(self, basic_info: dict) -> np.ndarray:
        return self.examination_encoder.encode_basic_info(basic_info)

    def sugar_7_point_encoder(self, seven_points_data: Dict) -> np.ndarray:
        # 七点血糖编码
        sugar_vec = []
        for point, data in seven_points_data.items():
            if data["measuring_blood_sugar"]:
                sugar = data["measuring_blood_sugar"]["value"]
            else:
                sugar = DEFAULT_MISSING_VALUE
            sugar_vec.append(sugar)
        return sugar_vec

    def sugar_24_point_encoder(
        self, seven_points_data: Dict, temporary_events: List[Dict] = None
    ) -> np.ndarray:
        # 将7点血糖做24点编码

        sugar_vec = [DEFAULT_MISSING_VALUE] * 24
        for point, data in seven_points_data.items():
            if data["measuring_blood_sugar"]:
                value = data["measuring_blood_sugar"]["value"]
                index = min(round(data["measuring_blood_sugar"]["time"]), 23)
                sugar_vec[index] = value
        for event in temporary_events:
            if event["action"] == "measuring_blood_sugar":
                index = min(round(event["time"]), 23)
                value = event["value"]
                sugar_vec[index] = value
        return sugar_vec

    def drug_1_point_encoder(self, seven_points_data: Dict) -> np.ndarray:
        drug_vec = self.drug_encoder.encode_drug_1_points(seven_points_data)
        return drug_vec

    def drug_7_point_encoder(self, seven_points_data: Dict) -> np.ndarray:
        drug_vec = self.drug_encoder.encode_drug_7_points(seven_points_data)
        return drug_vec

    def drug_24_point_encoder(self, seven_points_data: Dict) -> np.ndarray:
        drug_vec = self.drug_encoder.encode_drug_1_points(seven_points_data)
        drug_vec = [drug_vec] * 24
        return drug_vec

    def insulin_4_point_encoder(self, seven_points_data: Dict) -> np.ndarray:
        insulin_vec = []
        for point, data in seven_points_data.items():
            vec = [DEFAULT_MISSING_VALUE] * 9
            if int(point) % 2 == 0:
                for insulin in data["injecting_insulin"]:
                    insulin_name = clean_insulin_name(insulin["insulin"])
                    if insulin_name in ("胰岛素"):
                        continue
                    vec = self.insulin_encoder.encode_insulin(
                        insulin_name, insulin["value"]
                    )
                insulin_vec.append(vec)
        return insulin_vec

    def insulin_7_point_encoder(self, seven_points_data: Dict) -> np.ndarray:
        insulin_vec = []
        for point, data in seven_points_data.items():
            vec = [DEFAULT_MISSING_VALUE] * 9
            self.insulin_len_dc[len(data["injecting_insulin"])] += 1
            for insulin in data["injecting_insulin"]:
                insulin_name = clean_insulin_name(insulin["insulin"])
                if insulin_name in ("胰岛素"):
                    continue
                vec = self.insulin_encoder.encode_insulin(
                    insulin_name, insulin["value"]
                )
            insulin_vec.append(vec)
        return insulin_vec

    def insulin_24_point_encoder(
        self, seven_points_data: Dict, temporary_events: List[Dict] = None
    ) -> np.ndarray:
        insulin_vec = np.zeros((24, 9))
        insulin_vec.fill(DEFAULT_MISSING_VALUE)

        for point, data in seven_points_data.items():
            if int(point) % 2 == 0:
                for insulin in data["injecting_insulin"]:
                    insulin_name = clean_insulin_name(insulin["insulin"])
                    if insulin_name in ("胰岛素"):
                        continue
                    vec = self.insulin_encoder.encode_insulin(
                        insulin_name, insulin["value"]
                    )
                    index = min(round(insulin["time"]), 23)
                    insulin_vec[index] = vec
        for event in temporary_events:
            if event["action"] == "injecting_insulin":
                insulin_name = clean_insulin_name(event["insulin"])
                if insulin_name in ("胰岛素"):
                    continue
                vec = self.insulin_encoder.encode_insulin(insulin_name, event["value"])
                index = min(round(event["time"]), 23)
                insulin_vec[index] = vec
        return insulin_vec

    def temporary_insulin_7_point_encoder(self, temporary_events: List) -> np.ndarray:
        insulin_vec = np.zeros((7, 9))
        insulin_vec.fill(DEFAULT_MISSING_VALUE)
        for event in temporary_events:
            if event["action"] == "injecting_insulin":
                insulin_name = clean_insulin_name(event["insulin"])
                if insulin_name in ("胰岛素"):
                    continue
                vec = self.insulin_encoder.encode_insulin(insulin_name, event["value"])
                insulin_vec[point_24_to_7(event["time"])] = vec
        return insulin_vec

    def temporary_insulin_24_point_encoder(self, temporary_events: List) -> np.ndarray:
        insulin_vec = np.zeros((24, 9))
        insulin_vec.fill(DEFAULT_MISSING_VALUE)
        for event in temporary_events:
            if event["action"] == "injecting_insulin":
                insulin_name = clean_insulin_name(event["insulin"])
                if insulin_name in ("胰岛素"):
                    continue
                vec = self.insulin_encoder.encode_insulin(insulin_name, event["value"])
                index = min(round(event["time"]), 23)
                insulin_vec[index] = vec
        return insulin_vec

    def weather_encode(self, date_list: List[str]) -> np.ndarray:
        weather_vec = [
            self.weather_encoder.encode_weather_with_date(i) for i in date_list
        ]
        return np.array(weather_vec)
