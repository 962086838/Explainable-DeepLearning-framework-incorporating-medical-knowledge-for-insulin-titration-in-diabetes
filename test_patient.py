from typing import List
from datasets.patient import PatientInfo
from datasets.slices.flatten import slice_patient_info
# from datasets.dataset.mix_dynamic import MixDynamicPredictDataset
from datasets.dataset.mix_predict import MixPredictDataset
from datasets.utils import DatasetEncodeUtil
# from lt_mix_dynamic_nni import get_argument_parser, set_seed, MixDynamicPredictModule
from lt_mix_nni import get_argument_parser, set_seed, MixPredictModule

from lt_utils import get_loss_func
from utils.constants import PATIENT_BASE_DIR
from utils.np_loss import np_mae


# MODEL_FILE = "./.ckpts/mix_dynamic/epoch=24-val_loss=3.48-val_insulin_loss=1.25-val_sugar_loss=2.23.ckpt"
# MODEL_FILE = "./.ckpts/mix_model/epoch=34-val_loss=3.71-val_insulin_loss=1.23-val_sugar_loss=2.48.ckpt"

MODEL_FILE = "epoch=24-val_loss=3.48-val_insulin_loss=1.25-val_sugar_loss=2.23.ckpt"
# MODEL_FILE = "./.ckpts/mix_model/epoch=34-val_loss=3.71-val_insulin_loss=1.23-val_sugar_loss=2.48.ckpt"


SUGAR_MISS_MAXIMUM = 4
model_param = {
    "lr": 0.001,
    "optimizer_name": "adam",
    "lr_scheduler_name": "cosine_warm",
    "batch_size": 256,
    "max_feature_days": 3,
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

hp = get_argument_parser()

for k, v in model_param.items():
    hp[k] = v

set_seed(hp["seed"])



model = MixPredictModule.load_from_checkpoint(MODEL_FILE, hp=hp)
print(model.model.state_dict()["insulin_predict_dense.fc_2.bias"])

model.eval()
PatientInfo.base_dir = PATIENT_BASE_DIR
PatientInfo.util = DatasetEncodeUtil()


def test_patient_list(patient_id_list: List[str]):
    patient_list = [PatientInfo(f"{patient_id}.json") for patient_id in patient_id_list]

    test_dataset = MixPredictDataset(
        patient_list,
        max_feature_days=hp["max_feature_days"],
        min_feature_days=hp["min_feature_days"],
        sugar_miss_maximum=SUGAR_MISS_MAXIMUM,
    )
    x, (y_sugar, y_insulin) = test_dataset[:]
    # print(x.keys())  # dict_keys(['patient_id', 'examination_vec', 'day_vec_7_point', 'day_vec_24_point', 'point_vec_7_point', 'point_vec_24_point', 'sugar_vec_7_point', 'sugar_vec_24_point', 'drug_vec_1_point', 'drug_vec_7_point', 'drug_vec_24_point', 'insulin_vec_4_point', 'insulin_vec_7_point', 'insulin_vec_24_point', 'insulin_vec_7_point_temp', 'insulin_vec_24_point_temp'])

    predict_insulin, predict_sugar = model.forward(x)

    insulin_mae = get_loss_func("mae")(predict_insulin, y_insulin)
    sugar_mae = get_loss_func("mae")(predict_sugar, y_sugar)
    print(y_sugar.shape)
    print("patient_list: ", patient_id_list)
    print("insulin_mae: ", insulin_mae)
    print("sugar_mae: ", sugar_mae)

    # start = 0
    # end = -1
    # print(predict_insulin[start:end])
    # print(y_insulin[start:end])
    # print(get_loss_func("mae")(predict_insulin[start:end], y_insulin[start:end]))


test_patient_list(
    [
        # "1566729",
        # "1392750",
        # "1397635",
        # "1032809",
        # "1581403",
        # "874732",
        # "1140742",
        # "1093903",
        # "451342",
        # "862006",
        # "1587847",
        # "1496860",
        # "986641",
        # "1519067",
        '1466295',
    ]
)
