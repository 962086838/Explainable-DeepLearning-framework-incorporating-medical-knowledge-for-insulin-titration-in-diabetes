from lt_mix_dynamic_nni import *

set_seed(42)

model_param = {
    "lr": 0.001,
    "optimizer_name": "adam",
    "lr_scheduler_name": "cosine_warm",
    "batch_size": 1024,
    "max_feature_days": 9,
    "mlp_encode_dilation": 5,
    "mlp_output_dim": 256,
    "mlp_encode_blocks": 4,
    "mlp_encode_dropout": 0,
    "dense_hidden_size": 256,
    "dense_dropout": 0.3,
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

# hp['patient_limit'] = 100
# hp['sugar_miss_maximum'] = 6

model = MixDynamicPredictModule.load_from_checkpoint(
    "/mnt/sda1/libiao/models/mix/dynamic/mae/epoch=24-val_loss=3.48-val_insulin_loss=1.25-val_sugar_loss=2.23.ckpt",
    hp=hp,
)

# model.eval()

lt.Trainer(gpus=1).test(model)
