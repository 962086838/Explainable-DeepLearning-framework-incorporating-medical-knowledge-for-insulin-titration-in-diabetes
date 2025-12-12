import copy
from dis import dis
import math
import numpy as np
import pandas as pd
from datasets.patient import PatientInfo
from datasets.utils import (
    DrugEncoder,
    ExaminationEncoder,
    InsulinEncoder,
)
from utils.constants import DEFAULT_MISSING_VALUE
from IPython.core.display import display
import matplotlib.pyplot as plt
from utils.np_loss import np_mae, np_mare


def style_negative(v, n=-1, props=""):
    return props if v != n else None


def display_examination(array: np.ndarray, examination_encoder: ExaminationEncoder):
    print("\n检查信息:")
    eis = []
    for i, ex in enumerate(examination_encoder.examination_list):
        v = array[i]
        if v != DEFAULT_MISSING_VALUE:
            eis.append(ex)
            eis.append(v)
    eis = eis + [0] * (math.ceil(len(eis) / 8) * 8 - len(eis))
    eis = np.array(eis).reshape(-1, 8)
    df = pd.DataFrame(eis)
    df = replace_dataframe(df, -1.0, np.nan)
    df.columns = ["检查", "值"] * 4
    display(df)


def display_examination2(array: np.ndarray, examination_encoder: ExaminationEncoder):
    print("\n精简检查信息（性别 1 为男，0 为女）:")
    indexes = [
        "基本信息",
        "体格检查",
        "实验室检查1",
        "实验室检查2",
        "实验室检查3",
        "精氨酸实验1",
        "精氨酸实验2",
        "糖尿病相关抗体",
    ]

    def x_func(value, x):
        if value == -1:
            return "x"
        else:
            return round(value, x)

    data = [
        [
            "性别",
            f"{int(array[77])}",
            "年龄",
            f"{int(array[76])}",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            "身高",
            f"{x_func(array[75], 2)}",
            "体重",
            f"{x_func(array[1], 2)}",
            "BMI",
            "x",
            np.nan,
            np.nan,
        ],
        [
            "空腹血糖",
            f"{x_func(array[3], 2)}",
            "餐后2h血糖",
            f"{x_func(array[5], 2)}",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            "糖化白蛋白",
            f"{x_func(array[18], 2)}",
            "糖化血红蛋白",
            f"{x_func(array[2], 2)}",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            "丙氨酸氨基转移酶",
            f"{x_func(array[74], 2)}",
            "估算肾小球滤过率",
            f"{x_func(array[44], 2)}",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            "胰岛素",
            f"{x_func(array[19], 2)}",
            "胰岛素 2分钟",
            f"{x_func(array[10], 2)}",
            "胰岛素 4分钟",
            f"{x_func(array[11], 2)}",
            "胰岛素 6分钟",
            f"{x_func(array[16], 2)}",
        ],
        [
            "C肽",
            f"{x_func(array[33], 2)}",
            "C肽 2分钟",
            f"{x_func(array[14], 2)}",
            "C肽 4分钟",
            f"{x_func(array[15], 2)}",
            "C肽 6分钟",
            f"{x_func(array[9], 2)}",
        ],
        [
            "抗谷氨酸脱羧酶抗体",
            f"{x_func(array[68], 2)}",
            "抗胰岛素自身抗体",
            f"{x_func(array[69], 2)}",
            "抗胰岛细胞抗体",
            f"{x_func(array[70], 2)}",
            np.nan,
            np.nan,
        ],
    ]
    df = pd.DataFrame(data, index=indexes)
    # df = replace_dataframe(df, -1.0, np.nan)
    df = df.style.applymap(
        lambda x: "color:red;"
        if (isinstance(x, str) or not np.isnan(x)) and x != "x"
        else None
    )

    display(df)


def replace_dataframe(df: pd.DataFrame, origin, target):
    stack = df.stack()
    stack[stack == origin] = target
    df = stack.unstack()
    return df


def display_sugar(array: np.ndarray):
    print("\n血糖信息:")
    heads = ["早-前", "早-后", "中-前", "中-后", "晚-前", "晚-后", "睡-前"]
    index = [f"第{i+1}天" for i in range(0, array.shape[0])]
    df = pd.DataFrame(array, columns=heads, index=index)
    df = replace_dataframe(df, -1.0, np.nan)

    df = df.style.applymap(lambda x: "color:red;" if x > 0 else None).format("{:.2f}")
    display(df)


def display_sugar_24(array: np.ndarray):
    print("\n血糖信息:")
    heads = list(range(24))
    index = [f"第{i+1}天" for i in range(0, array.shape[0])]
    df = pd.DataFrame(array, columns=heads, index=index)
    df = replace_dataframe(df, -1.0, np.nan)

    df = df.style.applymap(lambda x: "color:red;" if x > 0 else None).format("{:.2f}")
    display(df)


def display_insulin(array, insulin_encoder: InsulinEncoder):
    print("\n胰岛素信息:")
    heads = ["早", "中", "晚", "睡"]
    insulin = []
    for i in range(array.shape[0]):
        insulin_day = dict()
        for j in range(array.shape[1]):
            index = int(array[i][j][0])
            if index != DEFAULT_MISSING_VALUE:
                name = insulin_encoder.insulin_list[index].name
                value = array[i][j][-1]
                insulin_day[heads[j]] = f"{name}: {value}"
            else:
                insulin_day[heads[j]] = -1
        insulin.append(insulin_day)

    index = [f"第{i+1}天" for i in range(0, array.shape[0])]
    df = pd.DataFrame(insulin, columns=heads, index=index)
    df = replace_dataframe(df, -1.0, np.nan)
    # df = df.style.applymap(lambda x: "color:red;" if isinstance(x, float) and not np.isnan(x) else None)
    df = df.style.applymap(
        lambda x: "color:red;" if isinstance(x, str) or not np.isnan(x) else None
    )
    display(df)


def display_drug(array: np.ndarray, drug_encoder: DrugEncoder):
    print("\n用药信息:")
    drug = list()
    for ds in array:
        cache = dict()
        for _d, d in enumerate(ds):
            if d != 0:
                cache[drug_encoder.drug_list[_d]] = d
        drug.append(cache)
    index = [f"第{i+1}天" for i in range(0, array.shape[0])]
    df = pd.DataFrame(drug, index=index)
    df = replace_dataframe(df, -1.0, np.nan)
    df = df.style.applymap(lambda x: "color:red;" if x > 0 else None).format("{:.2f}")
    display(df)


def display_sugar_compare_24(
    pi: PatientInfo,
    target_day,
    real_points: np.array,
    real_sugar: np.ndarray,
    human_predict: np.ndarray,
    model_predict: np.ndarray,
    baseline: np.ndarray,
    insulin: np.ndarray,
):
    print("\n血糖对比:")

    heads = real_points
    index = ["Insulin", "True", "Doctor", "Model", "Baseline"]
    array = np.stack([insulin, real_sugar, human_predict, model_predict, baseline])
    mae_loss = [
        0,
        0,
        np_mae(human_predict, real_sugar),
        np_mae(model_predict, real_sugar),
        np_mae(baseline, real_sugar),
    ]
    mare_loss = [
        0,
        0,
        np_mare(human_predict, real_sugar),
        np_mare(model_predict, real_sugar),
        np_mare(baseline, real_sugar),
    ]

    df = pd.DataFrame(array, columns=heads, index=index)
    plt_df = copy.deepcopy(df)
    plt_df[plt_df == -1] = np.nan
    plt_df.T.plot(figsize=(20, 10))

    df["mae"] = mae_loss
    df["mare"] = mare_loss
    df.to_csv(f"./pk_result/{pi.patient_id}_{target_day}.csv")
    df = df.style.applymap(style_negative, n=-1, props="color:red;").format("{:.2f}")
    display(df)


def display_sugar_compare(
    pi: PatientInfo,
    target_day,
    real_sugar: np.ndarray,
    human_predict: np.ndarray,
    model_predict: np.ndarray,
    baseline: np.ndarray,
):
    print("\n血糖对比:")
    heads = ["早-前", "早-后", "中-前", "中-后", "晚-前", "晚-后", "睡-前"]
    index = ["真实血糖", "医生预测血糖", "模型预测血糖1", "基准"]
    array = np.stack([real_sugar, human_predict, model_predict, baseline])
    mae_loss = [
        0,
        np_mae(human_predict, real_sugar),
        np_mae(model_predict, real_sugar),
        np_mae(baseline, real_sugar),
    ]
    mare_loss = [
        0,
        np_mare(human_predict, real_sugar),
        np_mare(model_predict, real_sugar),
        np_mare(baseline, real_sugar),
    ]

    df = pd.DataFrame(array, columns=heads, index=index)
    df["mae误差"] = mae_loss
    df["mare误差"] = mare_loss
    df.to_csv(f"./pk_result/{pi.patient_id}_{target_day}.csv")
    df = df.style.applymap(style_negative, n=-1, props="color:red;").format("{:.2f}")
    display(df)
