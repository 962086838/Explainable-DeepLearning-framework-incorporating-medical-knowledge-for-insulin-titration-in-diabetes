import os
import json
import pendulum
import numpy as np
import datetime as dt


def get_date_from_str(tm):
    if isinstance(tm, str):
        tm = dt.datetime.strptime(tm, "%Y-%m-%d %H:%M:%S")
    return tm.date()


def get_time_from_str(tm):
    if isinstance(tm, str):
        tm = dt.datetime.strptime(tm, "%Y-%m-%d %H:%M:%S")
    tm = tm.time()
    return float(tm.hour) + tm.minute / 60.0


def F_cmp(dic):
    tm = dic["time"]
    return dt.datetime.strptime(tm, "%Y-%m-%d %H:%M:%S")


def ordered(dict):
    new_dict = {}
    for key in sorted(dict):
        new_dict[key] = dict[key]
    return new_dict


def in_range(v, l, r):
    if l != -1 and v < l:
        return False
    elif r != -1 and v > r:
        return False
    return True


def preprocess_patient_info(patient_info: dict) -> dict:
    """
    TODO: 需要添加 将 patient info 处理为 模型模型接收的输入格式 的代码
    Parameters
    ----------
    patient_info : dict
        [description]

    Returns
    -------
    dict
        [description]
    """

    raw_version = example_to_raw(patient_info)  # raw
    data = save_in_timeline_form(raw_version)  # timeline_form
    data = simplify_timeline_form(data)  # simplified_timeline_form
    examination_data = take_out_first_day_examination_v2(data)  # first_day_examination
    data = save_in_timeline_form_without_examination(
        data
    )  # timeline_form_without_examination
    data = save_in_timeline_form_without_examination_and_other_drugs(
        data
    )  # timeline_form_without_examination_and_other_drugs
    data = v000(data, examination_data)
    data = fix_drug(data)
    data = v001(data)
    return data


def example_to_raw(data):

    raw_version = {
        "patient_id": data["patient_id"],
        "gender": int(data["gender"]),
        "patient__birth_date": data["birth_date"],
        "pre_labexam__": [],
        "pre_order__": [],
        "pre_physical_exam": [],
        "visit__admission_date": data["admission_time"],
        "visit__discharge_date": None,
    }

    # pre_labexam__
    for each in data["blood_sugar"]:
        raw_version["pre_labexam__"].append(
            {"n": "葡萄糖住院血糖报告", "v": each["value"], "t": each["check_time"], "u": None}
        )
    pre_physical_exam_name_list = ["体重", "收缩压", "血氧饱和度", "脉搏", "体温", "身高", "舒张压", "呼吸"]
    for each in data["examinations"]:
        if each["raw_name"] not in pre_physical_exam_name_list:
            raw_version["pre_labexam__"].append(
                {
                    "n": each["raw_name"],
                    "v": each["value"],
                    "t": each["check_time"],
                    "u": None,
                }
            )
    # pre_order__
    for each in data["drug"]:
        raw_version["pre_order__"].append(
            {
                "n": each["raw_name"],
                "nt": each["name_trade"],
                "planned_start_time": each["start_time"],
                "planned_end_time": each["end_time"],
                "prescribed_time": each["prescribed_time"],
                "order_status": each["order_status"],
                "performed_frequency": each["freq"],
                "v": each["dosage_per_use"],
                "u": each["unit"],
                "administration_method": "皮下" if each["is_insulin"] else "口服",  # 静注
                "long_term_or_temporary": "临时",  # "长期"
            }
        )
    # pre_physical_exam
    #  pre_physical_exam_name_list = ["体重", "收缩压", "血氧饱和度", "脉搏", "体温", "身高", "舒张压", "呼吸"]
    pre_physical_exam_dict = {
        "vital_sign_weight": [],
        "systolic_pressure": [],
        "pulse": [],
        "blood_pressure": [],
        "vital_sign_bmi": [],
        "temperature": [],
        "vital_sign_height": [],
        "diastolic_pressure": [],
        "respiration": [],
    }
    for each in data["examinations"]:
        if each["raw_name"] in pre_physical_exam_name_list:
            if each["raw_name"] == "体重":
                pre_physical_exam_dict["vital_sign_weight"].append(
                    {
                        "svalue": str(each["value"]),
                        "item_name": "体重",
                        "item_result_unit": "kg",
                        "check_time": each["check_time"],
                        "item_result": str(each["value"]),
                        "fvalue": each["value"],
                    }
                )
            elif each["raw_name"] == "收缩压":
                pre_physical_exam_dict["systolic_pressure"].append(
                    {
                        "svalue": str(each["value"]),
                        "item_name": "收缩压",
                        "item_result_unit": "mmHg",
                        "check_time": each["check_time"],
                        "ivalue": int(each["value"]),
                        "item_result": str(each["value"]),
                        "fvalue": float(each["value"]),
                    }
                )

            elif each["raw_name"] == "脉搏":
                pre_physical_exam_dict["pulse"].append(
                    {
                        "svalue": str(each["value"]),
                        "item_name": "脉搏",
                        "item_result_unit": "次/分",
                        "check_time": each["check_time"],
                        "ivalue": int(each["value"]),
                        "item_result": str(each["value"]),
                        "fvalue": float(each["value"]),
                    }
                )
            elif each["raw_name"] == "体温":
                pre_physical_exam_dict["temperature"].append(
                    {
                        "svalue": str(each["value"]),
                        "item_name": "体温",
                        "item_result_unit": "℃",
                        "check_time": each["check_time"],
                        "item_result": str(each["value"]),
                        "fvalue": float(each["value"]),
                    }
                )
            elif each["raw_name"] == "身高":
                pre_physical_exam_dict["vital_sign_height"].append(
                    {
                        "svalue": str(each["value"]),
                        "item_name": "身高",
                        "item_result_unit": "cm",
                        "check_time": each["check_time"],
                        "ivalue": int(each["value"]),
                        "item_result": str(each["value"]),
                        "fvalue": float(each["value"]),
                    }
                )
            elif each["raw_name"] == "舒张压":
                pre_physical_exam_dict["diastolic_pressure"].append(
                    {
                        "svalue": str(each["value"]),
                        "item_name": "舒张压",
                        "item_result_unit": "mmHg",
                        "check_time": each["check_time"],
                        "ivalue": int(each["value"]),
                        "item_result": str(each["value"]),
                        "fvalue": float(each["value"]),
                    }
                )
            elif each["raw_name"] == "呼吸":
                pre_physical_exam_dict["respiration"].append(
                    {
                        "svalue": str(each["value"]),
                        "item_name": "呼吸",
                        "item_result_unit": "次/分",
                        "check_time": each["check_time"],
                        "ivalue": int(each["value"]),
                        "item_result": str(each["value"]),
                        "fvalue": float(each["value"]),
                    }
                )
    raw_version["pre_physical_exam"] = [pre_physical_exam_dict]
    # json.dump(
    #     raw_version, open("../example_data/raw/tmp.json", "w"), ensure_ascii=False
    # )
    # # assert 1==0

    return raw_version


def save_in_timeline_form(data):
    pre_labexam__ = data["pre_labexam__"]
    pre_order__ = data["pre_order__"]
    pre_physical_exam = data["pre_physical_exam"][0]
    assert len(data["pre_physical_exam"]) == 1
    del data["pre_labexam__"]
    del data["pre_order__"]
    del data["pre_physical_exam"]

    timeline = []
    for x in pre_labexam__:
        if not x["v"] is None and not x["n"] is None and not x["t"] is None:
            x["time"] = x["t"]
            if float(x["v"]) == 0 and not x["n"] in [
                "酮体",
                "氨基末端利钠肽前体",
                "抗胰岛素自身抗体",
                "抗胰岛细胞抗体",
            ]:
                continue
            timeline.append(x)

    for x in pre_order__:
        if "v" not in x.keys() or x["v"] is None or float(x["v"]) == 0:
            continue
        if not x["planned_start_time"] is None:
            x["time"] = x["planned_start_time"]
        elif not x["t"] is None:
            x["time"] = x["t"]
        else:
            x["time"] = x["prescribed_time"]
        timeline.append(x)

    for x in (
        pre_physical_exam["respiration"]
        + pre_physical_exam["vital_sign_weight"]
        + pre_physical_exam["pulse"]
        + pre_physical_exam["temperature"]
        + pre_physical_exam["diastolic_pressure"]
        + pre_physical_exam["vital_sign_height"]
        + pre_physical_exam["systolic_pressure"]
    ):
        # print(x)
        if "fvalue" in x.keys() and "check_time" in x.keys():
            x["time"] = x["check_time"]
            if x["fvalue"] is None or float(x["fvalue"]) == 0:
                continue
            timeline.append(x)
    # print(timeline)
    timeline = sorted(timeline, key=lambda x: x["time"])
    data["timeline"] = timeline
    # json.dump(
    #     data, open(f"../example_data/timeline_form/tmp.json", "w"), ensure_ascii=False
    # )

    return data


def simplify_timeline_form(data):
    timeline = data["timeline"]
    # new_timeline = []
    for x in timeline:
        for old, new in zip(
            ["planned_end_time", "fvalue", "item_name", "item_result_unit"],
            ["time_end", "v", "n", "u"],
        ):
            if old in x.keys():
                x[new] = x[old]
                del x[old]
        for name in [
            "t",
            "planned_start_time",
            "planned_end_time",
            "order_status",
            "order_from",
            "svalue",
            "ivalue",
            "item_id",
            "item_result",
            "check_time",
        ]:
            if name in x.keys():
                del x[name]
        # new_timeline.append(x)

    # revise timeline
    ###deal with 白细胞计数
    for i, x in enumerate(timeline):
        if "n" in x.keys() and x["n"] == "白细胞计数":
            # xtm = dt.datetime.strptime(x["time"], "%Y-%m-%d %H:%M:%S")
            xtm = x["time"]
            find = False
            for j in range(i - 1, -1, -1):
                y = timeline[j]
                # ytm = dt.datetime.strptime(y["time"], "%Y-%m-%d %H:%M:%S")
                ytm = y["time"]
                if xtm - ytm < dt.timedelta(minutes=1):
                    # print(ytm, xtm)
                    if y["n"] == "颜色":
                        find = True
                        break
                else:
                    break
            for j in range(i + 1, len(timeline)):
                y = timeline[j]
                # ytm = dt.datetime.strptime(y["time"], "%Y-%m-%d %H:%M:%S")
                ytm = y["time"]
                if ytm - xtm < dt.timedelta(minutes=1):
                    # print(ytm, xtm)
                    if y["n"] == "颜色":
                        find = True
                        break
                else:
                    break
            if find:
                x["n"] = "白细胞计数(尿检)"

    data["timeline"] = timeline
    # json.dump(
    #     data,
    #     open(f"../example_data/simplified_timeline_form/tmp.json", "w"),
    #     ensure_ascii=False,
    # )
    # # exit()

    return data


def take_out_first_day_examination_v2(data):
    upper_lower_bound = json.load(
        open("./data/revise2-examinations_info_upper_lower_bound.json", "r")
    )["data"]

    timeline = data["timeline"]
    admission_date = get_date_from_str(data["visit__admission_date"])

    day0 = {}
    day1 = {}
    for x in timeline:
        if len(x.keys()) != 4 or "葡萄糖住院血糖报告" == x["n"]:
            continue
        # print(x)
        if x["n"] not in upper_lower_bound.keys() or not in_range(
            x["v"], upper_lower_bound[x["n"]]["min"], upper_lower_bound[x["n"]]["max"]
        ):
            continue
        x_date = get_date_from_str(x["time"])
        if x_date == admission_date:
            day0[x["n"]] = x["v"]
        if x_date == admission_date + dt.timedelta(days=1):
            day1[x["n"]] = x["v"]

    keys = set(day0.keys()).union(set(day1.keys()))
    all_examinations = {}
    for n in keys:
        v = 0
        c = 0
        for day in [day0, day1]:
            if n in day.keys():
                v += day[n]
                c += 1
        v = v / c
        all_examinations[n] = v

    addition_exam = [
        "C肽 空腹",
        "C肽 2分钟",
        "C肽 4分钟",
        "C肽 6分钟",
        "胰岛素 空腹",
        "胰岛素 2分钟",
        "胰岛素 4分钟",
        "胰岛素 6分钟",
    ]
    for key in addition_exam:
        if key in keys:
            continue
        result = None
        for x in timeline:
            if len(x.keys()) != 4 or "葡萄糖住院血糖报告" == x["n"]:
                continue
            if (
                x["n"] == key
                and x["v"] is not None
                and in_range(
                    x["v"], upper_lower_bound[key]["min"], upper_lower_bound[key]["max"]
                )
            ):
                result = x["v"]
                break

        if result is not None:
            all_examinations[key] = result

    examination_data = data.copy()
    del examination_data["timeline"]
    examination_data["day0_examination_len"] = len(day0.keys())
    examination_data["day0_examination"] = ordered(day0)
    examination_data["day1_examination_len"] = len(day1.keys())
    examination_data["day1_examination"] = ordered(day1)
    examination_data["all_examination_len"] = len(all_examinations.keys())
    examination_data["all_examination"] = ordered(all_examinations)
    # json.dump(
    #     examination_data,
    #     open(f"../example_data/first_day_examination/tmp.json", "w"),
    #     ensure_ascii=False,
    # )

    return examination_data
    # exit()


def save_in_timeline_form_without_examination(data):
    tmp = set()
    timeline = data["timeline"]
    new_timeline = []
    drugs = set()
    drugs2 = set()
    for x in timeline:
        if x["n"] == "葡萄糖住院血糖报告":
            assert len(x.keys()) == 4
            new_timeline.append(x)
        if len(x.keys()) != 4:
            if (
                x["n"] is None
            ):  # 手动处理 simplified_timeline_form 的 642296 795646 798000 835006 835037 873742 934595 100820?不太确定          818640醋酸泼尼松
                raise ValueError
            if x["n"] is not None:
                if x["v"] is not None:
                    drugs.add(x["n"])
                    new_timeline.append(x)
                if (
                    ("nt" in x.keys())
                    and (x["nt"] is not None)
                    and (x["v"] is not None)
                ):
                    drugs2.add(x["nt"])
                    if new_timeline == [] or new_timeline[-1] != x:
                        new_timeline.append(x)

            if x["n"] is None and "nt" in x.keys() and not x["nt"] is None:

                tmp.add(x["nt"])
    data["timeline"] = new_timeline
    data["drugs"] = sorted(list(drugs))
    data["drugs2"] = sorted(list(drugs2))

    return data


def save_in_timeline_form_without_examination_and_other_drugs(data):
    drugs = json.load(open(f"./data/needed_drugs.json", "r"))
    insulins = json.load(open(f"./data/all_insulins.json", "r"))
    timeline = data["timeline"]
    new_timeline = []
    his_drugs = set()
    for x in timeline:
        if x["n"] in ["葡萄糖住院血糖报告"]:
            new_timeline.append(x)
            continue
        if (
            "long_term_or_temporary" in x.keys()
            and x["long_term_or_temporary"] == "出院带药"
        ):
            continue
        if x["n"] in drugs:
            x["drug_id"] = drugs.index(x["n"])
            new_timeline.append(x)
            his_drugs.add(x["n"])
        elif x["n"] in insulins:
            x["insulin_id"] = insulins.index(x["n"])
            new_timeline.append(x)
            his_drugs.add(x["n"])
        elif x["nt"] in insulins:
            x["insulin_id"] = insulins.index(x["nt"])
            new_timeline.append(x)
            his_drugs.add(x["nt"])
        # else:
        #     raise ValueError(f"drug not found! [{x['n']}]")
    data["timeline"] = new_timeline
    data["drugs"] = sorted(list(his_drugs))
    return data


def v000(data, examination_data):
    drugs = json.load(open(f"./data/needed_drugs.json", "r"))
    all_examinations = json.load(open(f"./data/required_examinations.json", "r"))[
        "data"
    ]
    all_examination_keys = sorted(
        all_examinations.keys(),
        key=lambda k: (all_examinations[k]["importance"], all_examinations[k]["ratio"]),
        reverse=True,
    )

    drug_unit_statistics = [set() for _ in range(len(drugs))]
    new_data = {"patient_id": data["patient_id"]}

    admission_date = get_date_from_str(data["visit__admission_date"])
    birth_date = get_date_from_str(data["patient__birth_date"])
    basic_info = {
        "age": (admission_date - birth_date).days // 365,
        "gender": data["gender"],
    }
    for key in all_examination_keys:
        if key in examination_data["all_examination"].keys():
            basic_info[key] = examination_data["all_examination"][key]
    new_data["basic_info"] = basic_info

    timeline = data["timeline"]
    new_timeline = []
    for x in timeline:
        day = (get_date_from_str(x["time"]) - admission_date).days
        time = get_time_from_str(x["time"])
        tmp = None

        if x["n"] in drugs:
            drug_unit_statistics[drugs.index(x["n"])].add(x["u"])
            # if x['u'] is None and '阿卡波糖' in x['n']:
            #     print(patient_id, ' patient has None as unit in ', x['n'], x['v'])
            # if x['u'] is not None and '阿卡波糖' in x['n'] and float(x['v'])>=1:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '阿卡波糖' in x['n'] and x['v'] not in ['25.0', '50.0', '100.0', '0.05', '0.1']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '利拉鲁肽注射液' in x['n'] and x['v'] not in ['0.6', '1.2', '1.8']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '格列美脲片' in x['n']and x['v'] not in ['2.0', '4.0', '1.0', '3.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '注射用甲泼尼龙琥珀酸钠' in x['n']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '瑞格列奈片' in x['n'] and x['v'] not in ['0.5','1.0', '1.5', '2.0', '3.0', '4.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '甲泼尼龙片' in x['n'] and x['v'] not in ['8.0', '12.0', '10.0', '2.0', '3.0', '4.0', '16.0', '20.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '盐酸二甲双胍片' in x['n'] and x['v'] not in ['0.5', '0.85', '1.0', '1.7', '0.425']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '盐酸二甲双胍缓释片' in x['n'] and x['v'] not in ['0.25', '0.5', '1.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '盐酸吡格列酮片' in x['n'] and x['v'] not in ['15.0', '30.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '艾塞那肽注射液' in x['n'] and x['v'] not in ['5.0', '10.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '醋酸可的松片' in x['n'] and x['v'] not in ['5.0', '10.0', '12.5', '25.0', '50.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '醋酸地塞米松片' in x['n'] and x['v'] not in ['1.0', ]:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '醋酸泼尼松片' in x['n'] and x['v'] not in ['5.0','7.5',  '10.0', '15.0', '30.0', '60.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])
            # if '醋酸泼尼松龙片' in x['n'] and x['v'] not in ['2.5', '5.0', '7.5', '10.0', '15.0', '50.0', '60.0']:
            #     print(patient_id, ' patient has invalid value in ', x['n'], x['v'])

        """
        伏格列波糖片 {'mg'}     利拉鲁肽注射液 {'mg'}   利格列汀片 {'mg'}       格列吡嗪控释片 {'mg'}   格列吡嗪片 set()        
        格列喹酮片 {'mg'}       格列美脲片 {'mg'}       格列齐特片(II) {'mg'}   格列齐特缓释片 {'mg'}   沙格列汀片 {'mg'}       
        注射用甲泼尼龙琥珀酸钠 {'mg'}     瑞格列奈片 {'mg'}       甲泼尼龙片 {'mg'}       盐酸二甲双胍片 {'g'}    
        盐酸二甲双胍缓释片 {'g'}        盐酸吡格列酮片 {'mg'}   磷酸西格列汀片 {'mg'}   维格列汀片 {'mg'}       
        艾塞那肽注射液 set()    苯甲酸阿格列汀片 {'mg'} 西格列汀二甲双胍片 {'mg'}       那格列奈片 {'g'}  醋酸可的松片 {'mg'}     
        醋酸地塞米松片 {'mg'}   醋酸泼尼松片 {'mg'}     醋酸泼尼松龙片 {'mg'}   阿卡波糖片 {'mg', 'g'}  马来酸罗格列酮片 set()
        """

        if "葡萄糖住院血糖报告" == x["n"]:
            tmp = {
                "day": day,
                "time": time,
                "action": "measuring_blood_sugar",
                "value": float(x["v"]),
            }
        elif x["n"] in drugs:
            tmp = {
                "day": day,
                "time": time,
                "action": "taking_hypoglycemic_drugs",
                "drug": x["n"],
                "value": float(x["v"]),
                "day_end": (get_date_from_str(x["time_end"]) - admission_date).days,
                "time_end": get_time_from_str(x["time_end"]),
                "performed_frequency": x["performed_frequency"],
                "long_term_or_temporary": x["long_term_or_temporary"],
                "administration_method": x["administration_method"],
            }
        else:
            tmp = {
                "day": day,
                "time": time,
                "action": "injecting_insulin",
                "insulin": x["nt"],
                "value": float(x["v"]),
            }
        new_timeline.append(tmp)

    new_data["timeline"] = new_timeline

    return new_data


def fix_drug(data):

    # print(f"process {patient_id}")
    timeline = data["timeline"]
    for x in timeline:
        if (
            "action" in x.keys()
            and x["action"] == "injecting_insulin"
            and x["value"] > 201
        ):
            # print('before del', len(data['timeline']))
            timeline.remove(x)
            # print('after del', len(data['timeline']))
        elif "drug" in x.keys() and x["drug"] == "阿卡波糖片" and x["value"] > 300:
            timeline.remove(x)
        elif "drug" in x.keys() and x["drug"] == "阿卡波糖片" and 1e-5 < x["value"] < 25:
            x["value"] = x["value"] * 1000
            x["u"] = "mg"
        elif "drug" in x.keys() and x["drug"] == "醋酸地塞米松片" and x["value"] > 6:
            timeline.remove(x)
        elif "drug" in x.keys() and x["drug"] == "醋酸可的松片" and x["value"] > 100:
            timeline.remove(x)
        elif "drug" in x.keys() and x["drug"] == "利拉鲁肽注射液" and x["value"] > 1.8:
            timeline.remove(x)
        elif "drug" in x.keys() and x["drug"] == "盐酸吡格列酮片" and x["value"] > 45:
            timeline.remove(x)
        elif "drug" in x.keys() and x["drug"] == "盐酸二甲双胍缓释片" and x["value"] > 1.5:
            timeline.remove(x)
        elif "drug" in x.keys() and x["drug"] == "盐酸二甲双胍片" and x["value"] > 2.55:
            timeline.remove(x)
        elif (
            "drug" in x.keys() and x["drug"] == "注射用甲泼尼龙琥珀酸钠" and x["value"] < 10
        ):  # m, mg
            x["value"] = x["value"] * 1000
    # json.dump(
    #     data, open(f"../example_data/version000/tmp.json", "w"), ensure_ascii=False
    # )

    return data


def v001(data):
    times = [6, 8.5, 10.5, 13, 16.5, 19, 21]
    timeline = data["timeline"]
    longterm_drug = []
    mx_day = -1
    day_events = []

    for x in timeline:
        if x["action"] == "taking_hypoglycemic_drugs":
            if x["long_term_or_temporary"] == "长期" and (
                abs(x["day"] * 24 + x["time"] - x["day_end"] * 24 - x["time_end"]) > 1
            ):
                longterm_drug.append(x)
                continue

        while x["day"] > mx_day:
            mx_day += 1
            day_events.append(
                {
                    "measuring_blood_sugar": [],
                    "injecting_insulin": [],
                    "taking_hypoglycemic_drugs": [],
                }
            )
        # print(x["day"])
        if x["day"] >= 0:
            day_events[x["day"]][x["action"]].append(x)

    final_events = {}
    for i_day in range(mx_day + 1):
        events = day_events[i_day]
        temporary_events = []
        measuring_blood_sugar = events["measuring_blood_sugar"]
        tmp = []
        blood_sugar = [None for _ in range(7)]
        dp = [
            [
                {"cnt": 0, "diff": 0, "last": -1}
                for _ in range(len(measuring_blood_sugar) + 1)
            ]
            for _ in range(7 + 1)
        ]
        for x in range(1, len(times) + 1):
            for y in range(1, len(measuring_blood_sugar) + 1):
                if dp[x - 1][y]["cnt"] > dp[x][y]["cnt"] or (
                    dp[x - 1][y]["cnt"] == dp[x][y]["cnt"]
                    and dp[x - 1][y]["diff"] < dp[x][y]["diff"]
                ):
                    dp[x][y]["cnt"] = dp[x - 1][y]["cnt"]
                    dp[x][y]["diff"] = dp[x - 1][y]["diff"]
                    dp[x][y]["last"] = 0

                if dp[x][y - 1]["cnt"] > dp[x][y]["cnt"] or (
                    dp[x][y - 1]["cnt"] == dp[x][y]["cnt"]
                    and dp[x][y - 1]["diff"] < dp[x][y]["diff"]
                ):
                    dp[x][y]["cnt"] = dp[x][y - 1]["cnt"]
                    dp[x][y]["diff"] = dp[x][y - 1]["diff"]
                    dp[x][y]["last"] = 1

                if abs(measuring_blood_sugar[y - 1]["time"] - times[x - 1]) <= 1:
                    if dp[x - 1][y - 1]["cnt"] + 1 > dp[x][y]["cnt"] or (
                        dp[x - 1][y - 1]["cnt"] + 1 == dp[x][y]["cnt"]
                        and dp[x - 1][y - 1]["diff"]
                        + abs(measuring_blood_sugar[y - 1]["time"] - times[x - 1])
                        < dp[x][y]["diff"]
                    ):
                        dp[x][y]["cnt"] = dp[x - 1][y - 1]["cnt"] + 1
                        dp[x][y]["diff"] = dp[x - 1][y - 1]["diff"] + abs(
                            measuring_blood_sugar[y - 1]["time"] - times[x - 1]
                        )
                        dp[x][y]["last"] = 2

        x = len(times)
        y = len(measuring_blood_sugar)
        used = [False for _ in range(len(measuring_blood_sugar))]
        while x != 0 and y != 0:
            if dp[x][y]["last"] == 2:
                blood_sugar[x - 1] = measuring_blood_sugar[y - 1]
                used[y - 1] = True
                x -= 1
                y -= 1
            elif dp[x][y]["last"] == 1:
                y -= 1
            elif dp[x][y]["last"] == 0:
                x -= 1
            else:
                break

        for id, x in enumerate(measuring_blood_sugar):
            if not used[id]:
                temporary_events.append(x)

        insulin = [[] for _ in range(7)]
        for x in events["injecting_insulin"]:
            if x["time"] == 6:
                insulin[0].append(x)
            elif x["time"] == 10.5:
                insulin[2].append(x)
            elif x["time"] == 16.5:
                insulin[4].append(x)
            elif x["time"] == 21:
                insulin[6].append(x)
            else:
                temporary_events.append(x)

        drug = [[] for _ in range(7)]

        def in_time_range(d, t, d_s, t_s, d_e, t_e):
            _start_time = d_s * 24 + t_s
            _end_time = d_e * 24 + t_e
            _time = d * 24 + t
            if _start_time <= _time < _end_time:
                return True
            else:
                return False

        for x in events["taking_hypoglycemic_drugs"]:
            for i in range(6):
                if x["time"] >= times[i] and x["time"] < times[i + 1]:  # ???
                    drug[i].append(x)
            if x["time"] >= times[-1] or x["time"] < times[0]:
                drug[-1].append(x)

        for x in longterm_drug:
            t_id = []
            if x["performed_frequency"] == "qn":
                t_id = [6]
            elif x["performed_frequency"] == "tid":
                # t_id = [0, 2, 4]  # zhaohexu
                t_id = [1, 3, 5]
            elif x["performed_frequency"] == "bid":
                t_id = [0, 4]
            elif x["performed_frequency"] == "qd":
                t_id = [0]
            elif x["performed_frequency"] == "q8h":
                t_id = [0, 3, 6]
            elif x["performed_frequency"] == "q12h":
                t_id = [2, 6]
            elif x["performed_frequency"] == "qid":
                t_id = [0, 2, 4, 6]

            for id in t_id:
                if in_time_range(
                    i_day, times[id], x["day"], x["time"], x["day_end"], x["time_end"]
                ):
                    drug[id].append(x)

        temporary_events = sorted(temporary_events, key=lambda x: x["time"])
        today = {}
        for i in range(7):
            today[i] = {
                "measuring_blood_sugar": blood_sugar[i],
                "injecting_insulin": insulin[i],
                "taking_hypoglycemic_drugs": drug[i],
            }
        final_events[i_day] = {
            "daily_routine": today,
            "temporary_events": temporary_events,
        }

    del data["timeline"]
    data["days"] = final_events

    # json.dump(
    #     data, open(f"../example_data/version001/tmp.json", "w"), ensure_ascii=False
    # )

    return data


if __name__ == "__main__":
    tmp_json = json.load(open("../example_data/patient/840650(1).json", "r"))
    save_file_name = "patient_840650.json"
    save_path = os.path.join("../example_data/version001", save_file_name)
    # print(tmp_json)
    raw_version = example_to_raw(tmp_json)  # raw
    data = save_in_timeline_form(raw_version)  # timeline_form
    data = simplify_timeline_form(data)  # simplified_timeline_form
    examination_data = take_out_first_day_examination_v2(data)  # first_day_examination
    data = save_in_timeline_form_without_examination(
        data
    )  # timeline_form_without_examination
    data = save_in_timeline_form_without_examination_and_other_drugs(
        data
    )  # timeline_form_without_examination_and_other_drugs
    data = v000(data, examination_data)
    data = fix_drug(data)
    data = v001(data)

    preprocess_patient_info(tmp_json)
