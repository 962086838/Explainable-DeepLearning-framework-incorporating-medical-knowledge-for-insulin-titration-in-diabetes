import inspect
import random
import numpy as np

import pytorch_lightning as lt
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.dataset.sugar_predict import SugarPredictDataset
from datasets.patient import load_patient_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def filter_params_by_class(cl, params):
    model_params = inspect.signature(cl.__init__).parameters.keys()
    params = {k: v for k, v in params.items() if k in model_params}
    return params


def get_loss_func(loss_name: str):
    from utils import losses

    if loss_name == "rmse":
        loss = losses.rmse_loss
    elif loss_name == "weighted_rmse":
        loss = losses.weighted_rmse_loss
    elif loss_name == "rmse_ratio":
        loss = losses.rmse_ratio_loss
    elif loss_name == "smare":
        loss = losses.stepped_mean_absolutely_ratio_error
    elif loss_name == "mare":
        loss = losses.mean_absolutely_ratio_error
    elif loss_name == "mae":
        loss = losses.mean_absolutely_error
    return loss


class BaseDataMixin(lt.LightningModule):
    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp
        self.dataset_type: Dataset = None

    def prepare_data(self) -> None:
        # 目前还是用id读取，后续修改
        self.train_patient_list = load_patient_list(
            patient_filter_path="./resources/train_filename_list.txt",
            limit=self.hp["patient_limit"],
        )
        self.val_patient_list = load_patient_list(
            patient_filter_path="./resources/test_filename_list.txt",
            limit=self.hp["patient_limit"],
        )
        self.test_patient_list = load_patient_list(
            patient_filter_path="./resources/test_filename_list.txt",
            limit=self.hp["patient_limit"],
        )

        self.train_dataset = self.get_dataset_by_patient_list(self.train_patient_list)
        self.test_dataset = self.get_dataset_by_patient_list(self.test_patient_list)
        self.val_dataset = self.get_dataset_by_patient_list(self.val_patient_list)

    def get_dataset_by_patient_list(self, patient_list):
        return self.dataset_type(
            patient_list, **filter_params_by_class(self.dataset_type, self.hp)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hp["batch_size"], num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hp["batch_size"], num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hp["batch_size"], num_workers=0
        )
