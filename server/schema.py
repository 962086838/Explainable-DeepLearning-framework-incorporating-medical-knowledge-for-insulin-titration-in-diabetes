import datetime
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field
from utils.utils import random_json_data, read_json

from server.meta import InsulinPoint, InsulinType, Sugar7Point


def get_model_example_data(model: BaseModel):
    return model.Config.schema_extra["example"]


class ExamInfo(BaseModel):
    """
    体检信息
    """

    raw_name: str = Field(..., title="检查项名称")
    value: float = Field(0, title="值")
    check_time: datetime.datetime = Field(None, title="检查时间")


class BloodSugarInfo(BaseModel):
    """
    血糖信息
    """

    value: float = Field(0, title="血糖值")
    check_time: datetime.datetime = Field(None, title="检查时间")


class DrugInfo(BaseModel):
    """
    用药信息，血糖药和胰岛素
    """

    raw_name: str = Field(None, title="药品名称")
    name_trade: str = Field(None, title="商号")
    common_name: str = Field(None, title="通用名称")

    long_term_or_temporary: Optional[str] = Field(None, title="长期或者临时")
    start_time: datetime.datetime = Field(None, title="开始时间")
    end_time: datetime.datetime = Field(None, title="结束时间")
    prescribed_time: datetime.datetime = Field(None, title="规定时间")

    order_status: str = Field(None, title="状态")

    freq: str = Field(None, title="执行频率")
    dosage_per_use: float = Field(None, title="每次使用剂量")
    unit: str = Field(None, title="单位")

    is_insulin: bool = Field(False, title="是否为胰岛素")


class PatientInfo(BaseModel):
    patient_id: str = Field(..., title="病人ID", max_length=20)
    admission_time: datetime.datetime = Field(..., title="入院时间")
    gender: int = Field(..., title=" 性别， 0: 男，1: 女")
    birth_date: datetime.datetime = Field(
        ..., title="出生日期", description="为了统一时间格式，时间可以补0，类似 1968-08-01 00:00:00"
    )

    examinations: List[ExamInfo] = Field(None, title="检查数据")
    blood_sugar: List[BloodSugarInfo] = Field(None, title="血糖测量记录")
    drug: List[DrugInfo] = Field(None, title="用药信息")

    class Config:

        schema_extra = {"example": random_json_data("example_data/patient/")}


class InsulinInjection(BaseModel):
    inject_time: datetime.datetime = Field(..., title="注射时间")
    inject_point: InsulinPoint = Field(..., title="注射时间对应餐前餐后睡前时间")

    insulin_type: InsulinType = Field("", title="胰岛素类型")
    insulin_name: str = Field("", title="胰岛素名称")
    insulin_trade: str = Field("", title="胰岛素品牌")

    insulin_value: float = Field(..., title="注射剂量")

    class Config:
        schema_extra = {
            "example": {
                "inject_time": "2022-06-07T21:00:00",
                "inject_point": "bedtime",
                "insulin_type": InsulinType.basal,
                "insulin_name": "德谷胰岛素",
                "insulin_value": 10,
            }
        }


class InsulinRecommendResult(BaseModel):
    insulin_injection: List[InsulinInjection] = Field(None, title="胰岛素推荐注射")

    class Config:
        schema_extra = {
            "example": {"insulin": [get_model_example_data(InsulinInjection)]}
        }


class BloodSugar(BaseModel):
    point: Sugar7Point = Field(..., title="时间")
    value: float = Field(-1, title="血糖值")

    class Config:
        schema_extra = {"example": {"point": Sugar7Point.pre_morning, "value": 6.5}}


class BloodSugarResult(BaseModel):
    blood_sugar_list: List[BloodSugar] = Field(None, title="血糖列表")

    class Config:
        schema_extra = {
            "example": {
                "blood_sugar_list": [
                    BloodSugar(point=Sugar7Point.pre_morning, value=6),
                    BloodSugar(point=Sugar7Point.post_morning, value=10),
                    BloodSugar(point=Sugar7Point.pre_nooning, value=7),
                    BloodSugar(point=Sugar7Point.post_nooning, value=11),
                    BloodSugar(point=Sugar7Point.pre_evening, value=7.5),
                    BloodSugar(point=Sugar7Point.post_evening, value=10.5),
                    BloodSugar(point=Sugar7Point.bedtime, value=10),
                ]
            }
        }


class InsulinTarget(BaseModel):
    insulin_type: InsulinType = Field(..., title="胰岛素类型")
    insulin_name: str = Field(None, title="胰岛素名称", description="可指定需要打的胰岛素名称")
    insulin_trade: str = Field(None, title="胰岛素品牌", description="可指定需要打的胰岛素品牌")

    class Config:

        schema_extra = {
            "example": {
                "insulin_type": "shot",
                "insulin_name": "赖脯胰岛素注射液",
                "insulin_trade": "优泌乐",
            }
        }


# InsulinRecommend
# start-------
class InsulinRecommendRequest(BaseModel):
    insulin_target: List[InsulinTarget] = Field(
        ...,
        title="目标胰岛素类型",
        description="希望打的胰岛素种类，可以多种胰岛素一起使用",
    )
    start_point: InsulinPoint = Field(
        None, title="开始预测的时间点", description="开始预测的时间点，如果不输入，则按照上次血糖采样时间推测"
    )
    fake_now: datetime.datetime = Field(
        None, title="fake当前时间", description="因为有些病人距离目前很久远，调试的时候可以使用这个参数，造一个使用的时间"
    )
    patient_info: PatientInfo = Field(..., title="病人信息")

    class Config:
        schema_extra = {
            "example": {
                "insulin_target": [
                    InsulinTarget(
                        insulin_type="basal",
                        insulin_name="甘精胰岛素注射液",
                        insulin_trade="来得时 300u",
                    ),
                    InsulinTarget(
                        insulin_type="shot",
                        insulin_name="赖脯胰岛素注射液",
                        insulin_trade="优泌乐",
                    ),
                    InsulinTarget(
                        insulin_type="premix",
                        insulin_name="精蛋白锌重组赖脯胰岛素混合注射液",
                        insulin_trade="优泌乐25R 300u",
                    ),
                ],
                "start_point": InsulinPoint.morning,
                "fake_now": None,
                "patient_info": get_model_example_data(PatientInfo),
            }
        }


class InsulinRecommendResponse(BaseModel):
    record_id: str = Field(..., title="该条推荐记录的id")
    status: int = Field(0, title="状态", description="0为正常返回，非0则代表不同的保持。具体报错信息可参考msg")
    msg: str = Field("计算成功", title="参考消息")
    data: InsulinRecommendResult = Field(None, title="推荐结果")
    used_time: float = Field(0, title="消耗时间")

    class Config:
        schema_extra = {
            "example": {
                "record_id": str(uuid.uuid4()),
                "status": 0,
                "msg": "计算成功",
                "used_time": 1,
                "data": get_model_example_data(InsulinRecommendResult),
            }
        }


# InsulinRecommend
# end ----

# BloodSugarPredict
# start-------
class BloodSugarPredictRequest(BaseModel):
    expect_insulin: List[InsulinInjection] = Field(..., title="预计要打的胰岛素")
    start_point: Sugar7Point = Field(
        Sugar7Point.pre_morning, title="开始预测的时间点", description="开始预测的时间点，如果不输入，则按照上次血糖采样时间推测"
    )
    fake_now: datetime.datetime = Field(
        None, title="fake当前时间", description="因为有些病人距离目前很久远，调试的时候可以使用这个参数，造一个使用的时间"
    )
    
    patient_info: PatientInfo = Field(..., title="病人信息")
    insulin_record_id: str = Field(
        None,
        title="胰岛素推荐记录id",
        description="血糖预测是基于确定未来一天打的胰岛素剂量，所以会先推荐胰岛素剂量，给到对应胰岛素推荐的record_id，方便后续记录",
    )

    class Config:
        schema_extra = {
            "example": {
                "expect_insulin": [get_model_example_data(InsulinInjection)],
                "patient_info": get_model_example_data(PatientInfo),
                "start_point": Sugar7Point.pre_morning,
                "insulin_record_id": str(uuid.uuid4()),
            }
        }


class BloodSugarPredictResponse(BaseModel):
    record_id: str = Field("1233", title="该条推荐记录的id")
    status: int = Field(0, title="状态", description="0为正常返回，非0则代表不同的保持。具体报错信息可参考msg")
    msg: str = Field("计算成功", title="参考消息")
    used_time: float = Field(0, title="消耗时间")

    data: BloodSugarResult = Field(None, title="推荐结果")

    class Config:

        schema_extra = {
            "example": {
                "record_id": str(uuid.uuid4()),
                "status": 0,
                "msg": "计算成功",
                "used_time": 1,
                "data": get_model_example_data(BloodSugarResult),
            }
        }


# BloodSugarPredict
# end ----
