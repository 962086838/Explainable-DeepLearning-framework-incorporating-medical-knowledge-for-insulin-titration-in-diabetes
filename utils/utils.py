import json
import os
import random


def read_json(filename: str) -> dict:
    with open(filename, "r") as fin:
        return json.load(fin)


def random_json_data(path: str) -> dict:
    candidate_file = random.choice([i for i in os.listdir(path) if i.endswith(".json")])
    return read_json(os.path.join(path, candidate_file))


def transform_raw_to_examples(filename: str) -> dict:
    data = read_json(filename)
    result = {
        "patient_id": data["patient_id"],
        "gender": int(data["gender"]),
        "admission_time": data["visit__admission_date"],
        "birth_date": data["patient__birth_date"],
        "examinations": [],
        "blood_sugar": [],
        "drug": [],
    }
    for ex in data["pre_labexam__"]:
        if ex["n"] == "葡萄糖住院血糖报告":
            result["blood_sugar"].append({"value": ex["v"], "check_time": ex["t"]})
        else:
            if ex["n"]:
                result["examinations"].append(
                    {"raw_name": ex["n"], "value": ex["v"], "check_time": ex["t"]}
                )
    if data["pre_physical_exam"]:
        for k, exs in data["pre_physical_exam"][0].items():
            if not exs:
                continue
            ex = exs[0]
            if not isinstance(ex, dict):
                continue
            if ex["item_name"] and ex.get("svalue"):
                result["examinations"].append(
                    {
                        "raw_name": ex["item_name"],
                        "value": float(ex["svalue"]),
                        "check_time": ex["exam_time"],
                    }
                )
    for order in data["pre_order__"]:
        if "v" not in order.keys() or not order["n"]:
            continue
        _tmp_data = dict()
        _tmp_data["raw_name"] = order["n"]
        _tmp_data["name_trade"] = order.get("nt", "")
        _tmp_data["common_name"] = ""
        _tmp_data["start_time"] = order["planned_start_time"]
        _tmp_data["end_time"] = order["planned_end_time"]
        _tmp_data["prescribed_time"] = order["prescribed_time"]
        _tmp_data["order_status"] = order["order_status"]
        _tmp_data["freq"] = order["performed_frequency"]
        _tmp_data["dosage_per_use"] = float(order["v"])
        _tmp_data["unit"] = order["u"]
        _tmp_data["long_term_or_temporary"] = order.get("long_term_or_temporary")
        if _tmp_data.get("administration_method") == "皮下":
            _tmp_data["is_insulin"] = True
        else:
            _tmp_data["is_insulin"] = False
        result["drug"].append(_tmp_data)
    return result


def get_file_list():
    base_path = "/Users/liqiongyu/sqz/DataPreprocess/data/patients/raw/"
    file_list = [
        i for i in os.listdir("./example_data/patient/") if i.endswith(".json")
    ]
    if file_list:
        return [i.split("_")[1] for i in file_list]
    else:
        return [i for i in os.listdir(base_path) if i.endswith(".json")]


def batch_transform_raw_to_examples(limit: int = 10):
    base_path = "/Users/liqiongyu/sqz/DataPreprocess/data/patients/raw/"
    file_list = get_file_list()
    file_list = random.choices(file_list, k=limit) if limit else file_list
    for filename in file_list:
        try:
            data = transform_raw_to_examples(os.path.join(base_path, filename))
            with open(
                os.path.join("example_data/patient/", f"patient_{filename}"), "w"
            ) as fin:
                json.dump(data, fin, ensure_ascii=False)
        except Exception as e:
            print(filename)
            print(e)

def judge_plan_type(y_insulin):
    # tensor like torch.Size([1, 3, 7, 9])
    y_insulin = y_insulin[0, 2]
    plan_type = set()
    for each_time_dosage in y_insulin:
        if each_time_dosage[0] != -1:
            if each_time_dosage[1] == 0:
                plan_type.add("basal")
            elif each_time_dosage[1] == 1:
                plan_type.add("premix")
            elif each_time_dosage[1] == 2:
                plan_type.add("shot")
            else:
                raise ValueError
        # print(plan_type)
    plan_type = list(plan_type)
    if len(plan_type)>1 and "shot" in plan_type:
        plan_type.remove("shot")

    return sorted(plan_type)

def judge_plan_type_time(y_insulin):
    # tensor like torch.Size([1, 3, 7, 9])
    y_insulin = y_insulin[0, 2]
    plan_type = ""
    for _t, each_time_dosage in enumerate(y_insulin):
        if each_time_dosage[0] != -1:
            if each_time_dosage[1] == 0:
                plan_type+=f"t{_t}i0"
            elif each_time_dosage[1] == 1:
                plan_type+=f"t{_t}i1"
            elif each_time_dosage[1] == 2:
                plan_type+=f"t{_t}i2"
            else:
                raise ValueError

    return plan_type

