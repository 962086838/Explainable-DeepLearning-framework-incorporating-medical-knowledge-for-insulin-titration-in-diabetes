import inspect
import logging
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import nni
import numpy as np
import pytorch_lightning as lt
import torch
from nni.utils import merge_parameter
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor

from lt_utils import BaseDataMixin, get_loss_func, set_seed
from models.mix_predict_mask import MixPredictMaskModel
from datasets.dataset.mix_predict import MixPredictDataset

insulin_weight = 1
sugar_weight = 2


class MixPredictModule(BaseDataMixin):
    def __init__(self, hp: dict):
        super().__init__(hp)
        self.hp: dict = hp
        self.hp["window_size"] = hp["max_feature_days"]

        self.model_type = MixPredictMaskModel
        self.dataset_type = MixPredictDataset

        self.model = self.get_model()

    def get_model(self):
        model_params = inspect.signature(self.model_type).parameters.keys()
        params = {k: v for k, v in self.hp.items() if k in model_params}
        return self.model_type(**params)

    def configure_optimizers(self):
        # weight_decay = 1e-6  # l2正则化系数
        lr = self.hp["lr"]
        optim_name = self.hp["optimizer_name"]
        lr_scheduler_name = self.hp["lr_scheduler_name"]

        if optim_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError("未知 optimizer")

        if lr_scheduler_name == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.trainer.max_epochs, 0
            )
        elif lr_scheduler_name == "cosine_warm":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5
            )
        else:
            raise ValueError("未知 lr_scheduler")
        optim_dict = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return optim_dict

    def forward(self, x):
        insulin_data = self.model.unpack_insulin_data(x)
        sugar_data = self.model.unpack_sugar_data(x)

        insulin_output = self.model.insulin_forward(*insulin_data)
        sugar_output = self.model.sugar_forward(*sugar_data)

        return insulin_output, sugar_output

    def training_step(self, batch_data, batch_index) -> STEP_OUTPUT:
        x, (y_sugar, y_insulin) = batch_data
        insulin_out, sugar_out = self.forward(x)

        insulin_loss = self.calculate_loss(insulin_out, y_insulin)
        sugar_loss = self.calculate_loss(sugar_out, y_sugar)
        loss = insulin_loss * insulin_weight + sugar_loss * sugar_weight

        # result = {
        #     "train_insulin_loss": insulin_loss,
        #     "train_sugar_loss": sugar_loss,
        #     "train_step_loss": loss,
        # }
        # nni.report_intermediate_result(result)
        # self.log_dict(result, prog_bar=True)

        return {
            "loss": loss,
            "label_sugar": y_sugar.detach(),
            "label_insulin": y_insulin.detach(),
            "predict_sugar": sugar_out.detach(),
            "predict_insulin": insulin_out.detach(),
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        label_sugar = torch.cat([i["label_sugar"] for i in outputs])
        label_insulin = torch.cat([i["label_insulin"] for i in outputs])

        predict_sugar = torch.cat([i["predict_sugar"] for i in outputs])
        predict_insulin = torch.cat([i["predict_insulin"] for i in outputs])

        epoch_sugar_loss = self.calculate_loss(predict_sugar, label_sugar)
        epoch_insulin_loss = self.calculate_loss(predict_insulin, label_insulin)
        loss = epoch_sugar_loss + epoch_insulin_loss

        result = {
            "train_insulin_loss": epoch_insulin_loss,
            "train_sugar_loss": epoch_sugar_loss,
            "train_loss": loss,
        }
        # nni.report_intermediate_result(result)
        self.log_dict(result, prog_bar=True)

    def test_step(self, batch_data, batch_index) -> Tensor:
        x, (y_sugar, y_insulin) = batch_data
        insulin_out, sugar_out = self.forward(x)

        insulin_loss = self.calculate_loss(insulin_out, y_insulin)
        sugar_loss = self.calculate_loss(sugar_out, y_sugar)
        loss = insulin_loss * insulin_weight + sugar_loss * sugar_weight

        # result = {
        #     "test_insulin_loss": insulin_loss,
        #     "test_sugar_loss": sugar_loss,
        #     "test_step_loss": loss,
        # }
        # nni.report_intermediate_result(result)
        # self.log_dict(result, prog_bar=True)

        return {
            "loss": loss,
            "label_sugar": y_sugar.detach(),
            "label_insulin": y_insulin.detach(),
            "predict_sugar": sugar_out.detach(),
            "predict_insulin": insulin_out.detach(),
        }

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        label_sugar = torch.cat([i["label_sugar"] for i in outputs])
        label_insulin = torch.cat([i["label_insulin"] for i in outputs])

        predict_sugar = torch.cat([i["predict_sugar"] for i in outputs])
        predict_insulin = torch.cat([i["predict_insulin"] for i in outputs])

        epoch_sugar_loss = self.calculate_loss(predict_sugar, label_sugar)
        epoch_insulin_loss = self.calculate_loss(predict_insulin, label_insulin)
        loss = epoch_sugar_loss + epoch_insulin_loss

        result = {
            "test_insulin_loss": epoch_insulin_loss,
            "test_sugar_loss": epoch_sugar_loss,
            "test_loss": loss,
        }
        # nni.report_intermediate_result(result)
        nni.report_intermediate_result(
            {
                "default": float(loss),
                "test_insulin_loss": float(epoch_insulin_loss),
                "test_sugar_loss": float(epoch_sugar_loss),
            }
        )
        self.log_dict(result, prog_bar=True)

    def validation_step(self, batch_data, batch_index) -> Tensor:
        x, (y_sugar, y_insulin) = batch_data
        insulin_out, sugar_out = self.forward(x)

        insulin_loss = self.calculate_loss(insulin_out, y_insulin)
        sugar_loss = self.calculate_loss(sugar_out, y_sugar)
        loss = insulin_loss * insulin_weight + sugar_loss * sugar_weight

        # nni.report_intermediate_result(result)
        # self.log_dict(result, prog_bar=True)

        return {
            "loss": loss,
            "label_sugar": y_sugar.detach(),
            "label_insulin": y_insulin.detach(),
            "predict_sugar": sugar_out.detach(),
            "predict_insulin": insulin_out.detach(),
        }

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        label_sugar = torch.cat([i["label_sugar"] for i in outputs])
        label_insulin = torch.cat([i["label_insulin"] for i in outputs])

        predict_sugar = torch.cat([i["predict_sugar"] for i in outputs])
        predict_insulin = torch.cat([i["predict_insulin"] for i in outputs])

        epoch_sugar_loss = self.calculate_loss(predict_sugar, label_sugar)
        epoch_insulin_loss = self.calculate_loss(predict_insulin, label_insulin)
        loss = epoch_sugar_loss + epoch_insulin_loss

        result = {
            "val_insulin_loss": epoch_insulin_loss,
            "val_sugar_loss": epoch_sugar_loss,
            "val_loss": loss,
        }
        nni.report_intermediate_result(
            {
                "default": float(loss),
                "val_insulin_loss": float(epoch_insulin_loss),
                "val_sugar_loss": float(epoch_sugar_loss),
            }
        )
        self.log_dict(result, prog_bar=True)

    def calculate_loss(self, predict: Tensor, label: Tensor) -> Tensor:
        loss_func = get_loss_func(self.hp["loss_name"])
        loss = loss_func(predict, label)
        return loss

    def calculate_metrics(self, predict: Tensor, label: Tensor) -> Dict[str, Any]:
        pass


def get_save_base_dir(hp: Namespace):
    base_dir
    if "experiment_id" in hp and "trail_id" in hp:
        base_dir = ""


def main(hp: Namespace):
    set_seed(hp["seed"])
    model = MixPredictModule(hp)
    base_dir = os.path.join("/mnt/sda1/libiao/models/mix/", hp["loss_name"])

    trainer = lt.Trainer(
        gpus=hp["gpus"],
        max_epochs=hp["epochs"],
        strategy="ddp_spawn",
        default_root_dir=base_dir,
        # precision=16,
        # accumulate_grad_batches=4,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=20),
            ModelCheckpoint(
                dirpath=base_dir,
                save_top_k=2,
                save_last=True,
                monitor="val_loss",
                filename="{epoch}-{val_loss:.2f}-{val_insulin_loss:.2f}-{val_sugar_loss:.2f}",
            ),
        ],
    )
    trainer.fit(model)
    # nni.report_final_result(
    #     float(
    #         [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)][
    #             0
    #         ].best_model_score
    #     )
    # )
    trainer.test(model)
    trainer.save_checkpoint(os.path.join(base_dir, f"{hp['ckpt_name']}.ckpt"))


def get_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--ckpt_name", type=str, default="model")
    parser.add_argument("--loss_name", type=str, default="rmse")

    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--lr_scheduler_name", type=str, default="cosine")
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--max_feature_days", type=int, default=10)
    parser.add_argument("--min_feature_days", type=int, default=3)

    parser.add_argument("--patient_limit", type=int, default=20000)
    parser.add_argument("--sugar_miss_maximum", type=int, default=4)

    parser.add_argument("--mlp_encode_dilation", type=int, default=4)
    parser.add_argument("--mlp_output_dim", type=int, default=128)
    parser.add_argument("--mlp_encode_blocks", type=int, default=3)
    parser.add_argument("--mlp_encode_dropout", type=float, default=0.0001)

    parser.add_argument("--dense_hidden_size", type=int, default=128)
    parser.add_argument("--dense_dropout", type=float, default=0.5)

    parser.add_argument("--pool_name", type=str, default="max")
    parser.add_argument("--activation_name", type=str, default="relu")

    parser.add_argument("--attn_dim", type=int, default=1024)
    parser.add_argument("--attn_num_heads", type=int, default=8)
    parser.add_argument("--attn_dropout", type=float, default=0)

    parser.add_argument("--rnn_name", type=str, default="gru")
    parser.add_argument("--rnn_hidden_size", type=int, default=128)
    parser.add_argument("--rnn_num_layers", type=int, default=2)

    args = parser.parse_known_args()[0].__dict__
    return args


if __name__ == "__main__":
    try:
        args = get_argument_parser()
        tuner_params = nni.get_next_parameter()
        params = merge_parameter(args, tuner_params)
        params["experiment_id"] = nni.get_experiment_id()
        params["trail_id"] = nni.get_trial_id()
        print(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise e
