from typing import List
import numpy as np
import pandas as pd
from datasets.patient import PatientInfo
from datasets.slices.slice import slice_patient_info
from datasets.dataset.sugar_predict import mask_data_with_sugar
from datasets.utils import DatasetEncodeUtil
from jupyter_visual.display import (
    display_drug,
    display_examination,
    display_examination2,
    display_insulin,
    display_sugar,
    display_sugar_compare,
)
from lt_nni import SugarPredictModule, get_argument_parser, set_seed
from utils.constants import DEFAULT_MISSING_VALUE, PATIENT_BASE_DIR
from IPython.core.display import display, HTML
from IPython.display import clear_output


MAE_MODEL_FILE = "./.ckpts/epoch=37-val_epoch_loss=2.15.ckpt"
MAX_SUGAR_INPUT_TIMES = 3


model_param = {
    # "lr": 0.001,
    # "optimizer_name": "adam",
    # "lr_scheduler_name": "cosine",
    # "batch_size": 256,
    "max_feature_days": 9,
    "mlp_encode_dilation": 5,
    "mlp_output_dim": 256,
    "mlp_encode_blocks": 5,
    "mlp_encode_dropout": 0,
    "dense_hidden_size": 256,
    "dense_dropout": 0.30000000000000004,
    "pool_name": "max",
    "activation_name": "smu",
    "attn_dim": 512,
    "attn_num_heads": 8,
    "rnn_name": "gru",
    "rnn_hidden_size": 512,
    "rnn_num_layers": 2,
}


class StopException(Exception):
    pass


class PkUtil:
    def __init__(self):
        self.hp = get_argument_parser()
        for k, v in model_param.items():
            self.hp[k] = v
        set_seed(self.hp["seed"])
        self.model = SugarPredictModule.load_from_checkpoint(MAE_MODEL_FILE, hp=self.hp)
        self.model.eval()

        PatientInfo.base_dir = PATIENT_BASE_DIR
        PatientInfo.util = DatasetEncodeUtil()

    def pk_single_patient(self, patient_id: str, target_day: int = 3):
        clear_output()
        pi = PatientInfo(f"{patient_id}.json")
        test_data = mask_data_with_sugar(
            slice_patient_info(
                [pi],
                max_feature_days=self.hp["max_feature_days"],
                min_feature_days=self.hp["min_feature_days"],
            ),
            sugar_miss_maximum=7,
        )
        predict = self.model.forward(test_data[0]).detach().numpy()
        y = test_data[1].numpy()

        sugar_list = self.pk_single_patient_by_day(pi, target_day)

        print("你输入的血糖是: ", sugar_list)
        display_sugar_compare(
            pi,
            target_day,
            y[target_day - 3],
            np.array(sugar_list),
            predict[target_day - 3],
            y[target_day - 4],
        )

        return pi

    def pk_single_patient_by_day(self, pi: PatientInfo, day: int = 1):
        # clear_output()
        self.show_patient_info(pi, day)
        sugar_list = self.get_sugar_input(
            day,
            default_sugar_str=",".join(
                map(lambda x: str(x), pi.sugar_vec_7_point[day - 2])
            ),
        )

        return sugar_list

    def show_patient_info(self, pi: PatientInfo, day: int = 1):
        self.show_patient_examination(pi)
        self.show_patient_drug(pi, day)
        self.show_patient_insulin(pi, day)
        self.show_patient_sugar(pi, day - 1)

    def get_sugar_input(
        self, day: int = 3, default_sugar_str: str = "1,2,3,4,5,6,7"
    ) -> List[float]:
        for _ in range(3):
            try:
                input_text = input(
                    f"请输入第{day}天的7点血糖值，以英文逗号(,)分隔\
                    \n如 {default_sugar_str}\
                    \n如需要跳过该患者，请输入 stop:\n"
                )
                if input_text == "stop":
                    raise StopException()
                sugar_list = list(map(lambda x: float(x), input_text.split(",")))
                assert len(sugar_list) == 7, f"长度不符合，期望7，实际{len(sugar_list)}"
                return sugar_list
            except ValueError:
                print("输入有误！")
            except StopException as e:
                raise e
            except Exception as e:
                return
        else:
            print("输入错误超过3次")

    def show_patient_examination(self, pi: PatientInfo):
        display_examination(pi.examination_vec, pi.util.examination_encoder)
        display_examination2(pi.examination_vec, pi.util.examination_encoder)

    def show_patient_insulin(self, pi: PatientInfo, day: int = 3):
        display_insulin(pi.insulin_vec_4_point[:day], pi.util.insulin_encoder)

    def show_patient_sugar(self, pi: PatientInfo, day: int = 3):
        display_sugar(pi.sugar_vec_7_point[:day])

    def show_patient_drug(self, pi: PatientInfo, day: int = 3):
        display_drug(pi.drug_vec_1_point[:day], pi.util.drug_encoder)


def trans_table_str(x):
    return ",".join(x.split("\t"))
