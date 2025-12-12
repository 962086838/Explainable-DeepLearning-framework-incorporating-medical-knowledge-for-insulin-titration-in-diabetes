import datetime
import os
import time
from typing import List
import uuid

import structlog
from fastapi import FastAPI
from datasets.patient import PatientInfo
from datasets.utils import DatasetEncodeUtil
from server.example_to_version001 import preprocess_patient_info
from server.meta import InsulinPoint, InsulinType, Sugar7Point
from server.schema import (
    BloodSugar,
    BloodSugarResult,
    InsulinInjection,
    InsulinRecommendRequest,
    InsulinRecommendResponse,
    BloodSugarPredictRequest,
    BloodSugarPredictResponse,
    InsulinRecommendResult,
    get_model_example_data,
)
from fastapi.staticfiles import StaticFiles
from server.model_server import predict_util

structlog.configure(context_class=dict)
logger = structlog.get_logger()

app = FastAPI(
    title="Diabetes Management Apis",
    description="提供胰岛素推荐、血糖预测等借口",
    version="0.0.1",
)
PatientInfo.util = DatasetEncodeUtil()

if os.path.exists("/mnt/sda1/libiao/docker"):
    app.mount(
        "/static", StaticFiles(directory="/mnt/sda1/libiao/docker"), name="static"
    )

def dt_time_to_float(dt: datetime.datetime) -> float:
    # 将 datetime转化为float的时刻
    return dt.hour + dt.minute / 60


def float_time_to_point(dt: float) -> int:
    # 将float时间转为7点
    times = [6, 8.5, 10.5, 13, 16.5, 19, 21]
    for i in range(len(times) - 1):
        if times[i] < dt <= times[i + 1]:
            return i + 1
    return 0


def get_latest_time(patient_info: dict) -> str:
    dt1 = max([i["check_time"] for i in patient_info["blood_sugar"]])
    dt2 = max([i["start_time"] for i in patient_info["drug"]])
    return max([dt1, dt2])


def date_to_dict(dt) -> dict:
    return dict(year=dt.year, month=dt.month, day=dt.day)


# # 胰岛素类型对应常用胰岛素名称
# INSULIN_TYPE_DEFAULT_MAPPING = {
#     InsulinType.basal: {"name": "德谷胰岛素", "trade": ""},
#     InsulinType.shot: {"name": "德谷胰岛素", "trade": ""},
#     InsulinType.premix: {
#         "name": "(优泌乐)25-精蛋白锌重组赖脯胰岛素混合注射液",
#         "trade": "(优泌乐)25-精蛋白锌重组赖脯胰岛素混合注射液",
#     },
# }

# 胰岛素注射时间对应7点的index
INSULIN_POINT_TO_7_POINT_INDEX = {
    InsulinPoint.morning: 0,
    InsulinPoint.nooning: 2,
    InsulinPoint.evening: 4,
    InsulinPoint.bedtime: 6,
}

# 胰岛素类型对应需要注射的时间
INSULIN_TYPE_TO_POINT = {
    InsulinType.basal: [InsulinPoint.bedtime],
    InsulinType.premix: [InsulinPoint.morning, InsulinPoint.evening],
    InsulinType.shot: [
        InsulinPoint.morning,
        InsulinPoint.nooning,
        InsulinPoint.evening,
    ],
}


SUGAR_POINT_TO_INSULIN_POINT = {
    0: InsulinPoint.morning,
    1: InsulinPoint.morning,
    2: InsulinPoint.nooning,
    3: InsulinPoint.nooning,
    4: InsulinPoint.evening,
    5: InsulinPoint.evening,
    6: InsulinPoint.bedtime,
}


def get_insulin_time_by_point(
    today: datetime.date, start_point: InsulinPoint, target_point: InsulinPoint
):
    # times = [6, 8.5, 10.5, 13, 16.5, 19, 21]
    time_mapping = {
        "morning": {"hour": 6, "minute": 0, "second": 0},
        "nooning": {"hour": 10, "minute": 30, "second": 0},
        "evening": {"hour": 16, "minute": 30, "second": 0},
        "bedtime": {"hour": 21, "minute": 0, "second": 0},
    }

    if (
        INSULIN_POINT_TO_7_POINT_INDEX[target_point]
        >= INSULIN_POINT_TO_7_POINT_INDEX[start_point]
    ):  # 今日
        base_date = today
    else:  # 明日
        base_date = today + datetime.timedelta(days=1)

    return datetime.datetime(**date_to_dict(base_date), **time_mapping[target_point])


def pack_insulin_injection_list(
    insulin_target: List,
    out_dict: dict,
    dt_now: datetime.datetime,
    start_point: InsulinPoint,
) -> List[InsulinInjection]:
    """
    根据需要注射的胰岛素和胰岛素预测结果，拼接好服务返回结果。

    Parameters
    ----------
    insulin_target : List
        _description_
    out_dict : dict
        _description_
    dt_now : datetime.datetime
        _description_
    start_point : InsulinPoint
        _description_

    Returns
    -------
    List[InsulinInjection]
        _description_
    """
    insulin_list = []
    for it in insulin_target:
        # 通过胰岛素类型获取注射时间
        inject_point_list = INSULIN_TYPE_TO_POINT[it.insulin_type]
        # 根据注射时间 返回需要注射的胰岛素
        for inject_point in inject_point_list:
            insulin_list.append(
                InsulinInjection(
                    inject_time=get_insulin_time_by_point(
                        dt_now.date(), start_point, inject_point
                    ),
                    inject_point=inject_point,
                    insulin_type=it.insulin_type,
                    insulin_name=it.insulin_name,
                    insulin_trade=it.insulin_trade,
                    insulin_value=out_dict[
                        INSULIN_POINT_TO_7_POINT_INDEX[inject_point]
                    ],
                )
            )
    return insulin_list


def insert_insulin_to_patient_drugs(drug_list: list, expect_insulin: list):
    for insulin in expect_insulin:
        print("insert insulin:", insulin["insulin_name"], insulin["insulin_value"])
        drug_list.append(
            {
                "raw_name": insulin["insulin_name"],
                "name_trade": insulin["insulin_trade"],
                "common_name": "",
                "long_term_or_temporary": "临时",
                "start_time": insulin["inject_time"],
                "end_time": insulin["inject_time"] + datetime.timedelta(seconds=1),
                "prescribed_time": insulin["inject_time"] - datetime.timedelta(days=1),
                "order_status": "已停止",
                "freq": "st",
                "dosage_per_use": insulin["insulin_value"],
                "unit": "IU",
                "is_insulin": True,
            }
        )
    return drug_list


@app.post("/v2/insulin_recommend/", response_model=InsulinRecommendResponse)
async def insulin_recommend(body: InsulinRecommendRequest):
    t1 = time.time()
    # 获取当前时间
    if body.fake_now:
        dt_now = body.fake_now
    else:
        dt_now = datetime.datetime.now()

    patient_info = body.dict()["patient_info"]
    patient_dt_latest = get_latest_time(patient_info)
    print("patient_dt_latest: ", patient_dt_latest)

    # 获取开始预测时间点
    if body.start_point:
        start_point = body.start_point
    else:
        start_point = SUGAR_POINT_TO_INSULIN_POINT[
            float_time_to_point(dt_time_to_float(dt_now))
        ]
    print("start_point:", start_point)

    # if (dt_now - patient_dt_latest).days > 10:
    #     raise Exception("该病人间隔时间超过十天")

    # 预处理数据
    # 会给向量padding对应的值。
    # todo: 还差一个讲target insulin 嵌入到模型的处理过程
    patient_info = preprocess_patient_info(patient_info)
    pi = PatientInfo(patient_data=patient_info)
    data = predict_util.preprocess_data(pi, INSULIN_POINT_TO_7_POINT_INDEX[start_point])

    print("sugar-----")
    print(pi.sugar_vec_7_point)
    print(data["sugar_vec_7_point"][0].reshape(9, 7).detach().numpy())

    print("insulin_type-----")
    print(pi.insulin_vec_7_point[:, :, 1])
    print(data["insulin_vec_7_point"][0][:, 1].reshape(9, 7).detach().numpy())

    print("insulin_value-----")
    print(pi.insulin_vec_7_point[:, :, -1])
    print(data["insulin_vec_7_point"][0][:, -1].reshape(9, 7).detach().numpy())

    # 获取推荐结果，此处可能需要重写
    out = predict_util.insulin_predict(data)[0]

    # 组合推荐结果
    point = (
        data["point_vec_7_point"][0]
        .reshape(predict_util.max_feature_days, 7)
        .detach()
        .numpy()[-1]
    )
    out_dict = {int(k): round(v, 2) for k, v in zip(point, out)}
    print(out_dict)

    insulin_list = pack_insulin_injection_list(
        body.insulin_target, out_dict, dt_now, start_point
    )
    # 推荐结果按时间排序
    insulin_list.sort(key=lambda x: x.inject_time)
    rp = InsulinRecommendResponse(
        record_id=str(uuid.uuid4()),
        data=InsulinRecommendResult(insulin_injection=insulin_list),
        used_time=time.time() - t1,
    )
    return rp


@app.post("/v2/blood_sugar_predict/", response_model=BloodSugarPredictResponse)
async def blood_sugar_predict(body: BloodSugarPredictRequest):
    t1 = time.time()

    # 获取当前时间
    if body.fake_now:
        dt_now = body.fake_now
    else:
        dt_now = datetime.datetime.now()

    # 获取开始预测时间点
    if body.start_point:
        start_point = body.start_point
    else:
        start_point = SUGAR_POINT_TO_INSULIN_POINT[
            float_time_to_point(dt_time_to_float(dt_now))
        ]
    print("start_point:", start_point)

    data = body.dict()
    expect_insulin = data["expect_insulin"]
    patient_info = data["patient_info"]

    patient_dt_latest = get_latest_time(patient_info)
    print("patient_dt_latest: ", patient_dt_latest)

    # # 获取之前注射过的胰岛素，未注射过的胰岛素不支持
    # insulin_candidate = copy.deepcopy(INSULIN_TYPE_DEFAULT_MAPPING)

    # for drug in patient_info["drug"]:
    #     insulin_name = drug["name_trade"]
    #     insulin = PatientInfo.util.insulin_encoder.insulin_mapping.get(insulin_name)
    #     if insulin:
    #         insulin_candidate[insulin.classification] = insulin_name
    # with open("./tempo1.json", "w") as fin:
    #     json.dump(patient_info, fin, default=str, ensure_ascii=False)

    # with open("./temp.json", "w") as fin:
    #     json.dump(
    #         preprocess_patient_info(patient_info), fin, default=str, ensure_ascii=False
    #     )

    # pi = PatientInfo(patient_data=preprocess_patient_info(patient_info))
    # # patient_dt_latest = get_latest_time(patient_info)
    # # print(patient_dt_latest)
    # print(pi.max_days)

    # for insulin in expect_insulin:
    #     print("insert insulin:", insulin["insulin_name"], insulin["insulin_value"])
    #     patient_info["drug"].append(
    #         {
    #             "raw_name": insulin["insulin_name"],
    #             "name_trade": insulin["insulin_trade"],
    #             "common_name": "",
    #             "long_term_or_temporary": "临时",
    #             "start_time": insulin["inject_time"],
    #             "end_time": insulin["inject_time"] + datetime.timedelta(seconds=1),
    #             "prescribed_time": insulin["inject_time"] - datetime.timedelta(days=1),
    #             "order_status": "已停止",
    #             "freq": "st",
    #             "dosage_per_use": insulin["insulin_value"],
    #             "unit": "IU",
    #             "is_insulin": False,
    #         }
    #     )
    # 加入期望注射的insulin
    drug_list = insert_insulin_to_patient_drugs(patient_info["drug"], expect_insulin)
    drug_list.sort(key=lambda x: x["start_time"])
    patient_info["drug"] = drug_list
    # with open("./tempo2.json", "w") as fin:
    #     json.dump(patient_info, fin, default=str, ensure_ascii=False)

    # with open("./temp2.json", "w") as fin:
    #     json.dump(
    #         preprocess_patient_info(patient_info), fin, default=str, ensure_ascii=False
    #     )

    print("admission_time:", patient_info["admission_time"])
    # 预处理病人信息
    patient_info = preprocess_patient_info(patient_info)
    pi = PatientInfo(patient_data=patient_info)
    data = predict_util.preprocess_data_sugar(
        pi, start_point=list(Sugar7Point).index(start_point)
    )
    print(pi.max_days)

    print("-----")
    print(pi.sugar_vec_7_point)
    print(data["sugar_vec_7_point"][0].reshape(9, 7).detach().numpy())

    print("-----")
    print(pi.insulin_vec_7_point[:, :, -1])
    print(data["insulin_vec_7_point"][0][:, -1].reshape(9, 7).detach().numpy())

    print("-----")
    print(pi.insulin_vec_7_point[:, :, 1])
    print(data["insulin_vec_7_point"][0][:, 1].reshape(9, 7).detach().numpy())

    # 预测血糖
    sugar = predict_util.sugar_predict(data)[0]
    point = (
        data["point_vec_7_point"][0]
        .reshape(predict_util.max_feature_days, 7)
        .detach()
        .numpy()[-1]
    )

    out_dict = {int(k): v for k, v in zip(point, sugar)}
    print(out_dict)

    blood_sugar_list = []
    for p, s in zip(point, sugar):
        blood_sugar_list.append(BloodSugar(point=list(Sugar7Point)[int(p)], value=s))

    rp = BloodSugarPredictResponse(record_id=str(uuid.uuid4()))
    rp.data = BloodSugarResult(blood_sugar_list=blood_sugar_list)
    rp.used_time = time.time() - t1
    return rp
