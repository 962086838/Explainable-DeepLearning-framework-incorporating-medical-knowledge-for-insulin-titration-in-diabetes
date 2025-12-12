import inspect
import numpy as np
import torch
from datasets.patient import PatientInfo
from models.mix_predict_dynamic import MixPredictDynamicModel
from keras_preprocessing.sequence import pad_sequences

from utils.constants import DAY_DIM, DRUG_DIM, INSULIN_DIM, POINT_DIM, SUGAR_DIM

CKPT_PATH = "./.ckpts/mix_model/latest.ckpt"
m_config = {
    "lr": 0.001,
    "optimizer_name": "adam",
    "lr_scheduler_name": "cosine_warm",
    "batch_size": 256,
    "max_feature_days": 9,
    "mlp_encode_dilation": 5,
    "mlp_output_dim": 256,
    "mlp_encode_blocks": 4,
    "mlp_encode_dropout": 0,
    "dense_hidden_size": 256,
    "dense_dropout": 0.30000000000000004,
    "pool_name": "avg",
    "activation_name": "prelu",
    "attn_dim": 256,
    "attn_num_heads": 2,
    "rnn_name": "gru",
    "rnn_hidden_size": 512,
    "rnn_num_layers": 2,
}


class MixPredictUtil:
    def __init__(self):
        self.max_feature_days: int = 10
        self.min_feature_days: int = 3
        self.point = 7
        self.model: MixPredictDynamicModel = None
        self._load_model()

    def _load_model(self):
        params = m_config

        self.max_feature_days = params["max_feature_days"]
        # self.min_feature_days = params["min_feature_days"]
        params["window_size"] = params["max_feature_days"]

        params = {
            k: v
            for k, v in params.items()
            if k in inspect.signature(MixPredictDynamicModel).parameters.keys()
        }
        model = MixPredictDynamicModel(**params)
        model.load_state_dict(torch.load(CKPT_PATH, map_location=torch.device("cpu")))
        model.eval()
        self.model = model

    def predict(self):
        pass

    def post_padding_data(self, patient: PatientInfo):
        # 往后 padding 一天的信息，不同数据填充方式不一样
        # 这样做是为了在通用模型下能预测未来7个点的数据
        max_days = patient.max_days + 1
        total_point = max_days * 7

        def _post_padding(vec):
            return pad_sequences(
                [vec], maxlen=max_days, dtype="float32", value=-1, padding="post"
            )

        result = dict()

        result["day_vec"] = (
            np.arange(max_days)
            .reshape(-1, 1)
            .repeat(7, axis=1)
            .reshape(total_point, DAY_DIM)
        )
        result["point_vec_7_point"] = (
            np.arange(7)
            .reshape(1, 7)
            .repeat(max_days, axis=0)
            .reshape(total_point, POINT_DIM)
        )

        # 血糖胰岛素直接填充-1
        result["sugar_vec_7_point"] = _post_padding(patient.sugar_vec_7_point).reshape(
            total_point, SUGAR_DIM
        )
        result["insulin_vec_7_point"] = _post_padding(
            patient.insulin_vec_7_point
        ).reshape(total_point, INSULIN_DIM)

        result["insulin_vec_7_point_temp"] = _post_padding(
            patient.insulin_vec_7_point_temp
        ).reshape(total_point, INSULIN_DIM)

        # drug 信息填充前一天
        result["drug_vec_7_point"] = np.concatenate(
            [patient.drug_vec_7_point, patient.drug_vec_7_point[-1:]]
        ).reshape(total_point, DRUG_DIM)
        return result

    def preprocess_data(self, patient: PatientInfo, start_point: int = 0):
        end = start_point - 7
        vec_data = self.post_padding_data(patient)

        def _pad(_vec):
            return pad_sequences(
                [_vec],
                maxlen=self.max_feature_days * self.point,
                dtype="float32",
                value=-1,
                padding="pre",  # 不够的往前补-1
                truncating="pre",  # 太长了往前截断
            )

        day_vec = _pad(vec_data["day_vec"][:end])
        point_vec_7_point = _pad(vec_data["point_vec_7_point"][:end])

        sugar_vec_7_point = _pad(vec_data["sugar_vec_7_point"][:end])
        drug_vec_7_point = _pad(vec_data["drug_vec_7_point"][:end])
        insulin_vec_7_point = _pad(vec_data["insulin_vec_7_point"][:end])
        insulin_vec_7_point_temp = _pad(vec_data["insulin_vec_7_point_temp"][:end])

        data = {
            "examination_vec": torch.FloatTensor([patient.examination_vec]),
            "day_vec_7_point": torch.FloatTensor(day_vec),
            "point_vec_7_point": torch.FloatTensor(point_vec_7_point),
            "sugar_vec_7_point": torch.FloatTensor(sugar_vec_7_point),
            "drug_vec_7_point": torch.FloatTensor(drug_vec_7_point),
            "insulin_vec_7_point": torch.FloatTensor(insulin_vec_7_point),
            "insulin_vec_7_point_temp": torch.FloatTensor(insulin_vec_7_point_temp),
        }
        return data

    def preprocess_data_sugar(self, patient: PatientInfo, start_point: int = 0):
        if start_point == 0:
            end = None
        else:
            end = start_point - 7
        # end = 10000
        max_days = patient.max_days
        total_point = max_days * 7

        def _pad(_vec):
            return pad_sequences(
                [_vec],
                maxlen=self.max_feature_days * self.point,
                dtype="float32",
                value=-1,
            )

        day_vec = _pad(patient.day_vec_7_point.reshape(total_point, DAY_DIM)[:end])
        point_vec_7_point = _pad(
            patient.point_vec_7_point.reshape(total_point, POINT_DIM)[:end]
        )

        sugar_vec_7_point = _pad(
            patient.sugar_vec_7_point.reshape(total_point, SUGAR_DIM)[:end]
        )
        insulin_vec_7_point = _pad(
            patient.insulin_vec_7_point.reshape(total_point, INSULIN_DIM)[:end]
        )
        insulin_vec_7_point_temp = _pad(
            patient.insulin_vec_7_point_temp.reshape(total_point, INSULIN_DIM)[:end]
        )
        drug_vec_7_point = _pad(patient.drug_vec_7_point.reshape(total_point, DRUG_DIM)[:end])

        data = {
            "examination_vec": torch.FloatTensor([patient.examination_vec]),
            "day_vec_7_point": torch.FloatTensor(day_vec),
            "point_vec_7_point": torch.FloatTensor(point_vec_7_point),
            "sugar_vec_7_point": torch.FloatTensor(sugar_vec_7_point),
            "drug_vec_7_point": torch.FloatTensor(drug_vec_7_point),
            "insulin_vec_7_point": torch.FloatTensor(insulin_vec_7_point),
            "insulin_vec_7_point_temp": torch.FloatTensor(insulin_vec_7_point_temp),
        }
        return data

    def sugar_predict(self, data):
        output = self.model.sugar_forward(*self.model.unpack_sugar_data(data))
        return output.detach().numpy()

    def insulin_predict(self, data):
        output = self.model.insulin_forward(*self.model.unpack_insulin_data(data))
        return output.detach().numpy()


predict_util = MixPredictUtil()
