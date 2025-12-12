import inspect

import torch

# 这个是mix predict用的模型
from models.mix_predict_mask import MixPredictMaskModel

# 这个是mix predict 用的dataset
from datasets.dataset.mix_predict import MixPredictDataset


# 这个是lightning使用的模型文件路径
CKPT_PATH = (
    "h1:/mnt/sda1/libiao/models/practicable_models/mix_predict_lightning_model.ckpt"
)

# 这个是pytorch使用的模型文件路径
CKPT_PATH = (
    "h1:/mnt/sda1/libiao/models/practicable_models/mix_predict_pytorch_model.ckpt"
)

# 示例路径，直接load pytorch 文件
CKPT_PATH = "./.ckpts/mix_model/latest.ckpt"

params = {
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

params["window_size"] = params["max_feature_days"]

params = {
    k: v
    for k, v in params.items()
    if k in inspect.signature(MixPredictMaskModel).parameters.keys()
}
model = MixPredictMaskModel(**params)
model.load_state_dict(torch.load(CKPT_PATH, map_location=torch.device("cpu")))
