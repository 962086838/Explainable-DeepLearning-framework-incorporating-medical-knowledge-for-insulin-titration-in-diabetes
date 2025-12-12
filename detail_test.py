import os
import random
import inspect
from typing import Any, Dict
import numpy as np
# import pytorch_lightning as lt
from torch import Tensor
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from datasets.patient import load_patient_list
from models.mix_predict_mask import MixPredictMaskModel
from datasets.dataset.mix_predict import MixPredictDataset, MixPredictDataset_targetpidday
import shapley_utils
from lt_mix_nni import get_argument_parser, set_seed, MixPredictModule

from utils.constants import PATIENT_BASE_DIR
from datasets.patient import PatientInfo
from datasets.utils import DatasetEncodeUtil

'''

'''


from tqdm import tqdm

from shapley_taylor import ShapleyTaylor, GroupShapleyTaylor, GroupKernelShapleyTaylor

insulin_weight = 1
sugar_weight = 2




def main(hp: Namespace):
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
    # father_model = SugarPredictModule(hp)
    CKPT_PATH = "./epoch=24-val_loss=3.48-val_insulin_loss=1.25-val_sugar_loss=2.23.ckpt"
    father_model = MixPredictModule.load_from_checkpoint(CKPT_PATH, hp=hp)
    # father_model = MixPredictModule(hp=hp)
    # base_dir = os.path.join(hp.base_dir, hp.loss_name)
    father_model.eval()
    # params = {
    #     "lr": 0.001,
    #     "optimizer_name": "adam",
    #     "lr_scheduler_name": "cosine_warm",
    #     "batch_size": 256,
    #     "max_feature_days": 3,  # 3 9
    #     "mlp_encode_dilation": 5,
    #     "mlp_output_dim": 256,
    #     "mlp_encode_blocks": 4,
    #     "mlp_encode_dropout": 0,
    #     "dense_hidden_size": 256,
    #     "dense_dropout": 0.30000000000000004,
    #     "pool_name": "avg",
    #     "activation_name": "prelu",
    #     "attn_dim": 256,
    #     "attn_num_heads": 2,
    #     "rnn_name": "gru",
    #     "rnn_hidden_size": 512,
    #     "rnn_num_layers": 2,
    # }
    # for k, v in params.items():
    #     hp[k] = v
    # params["window_size"] = params["max_feature_days"]
    # CKPT_PATH = "./mix_predict_pytorch_model.ckpt"
    # CKPT_PATH = "./epoch=34-val_loss=3.71-val_insulin_loss=1.23-val_sugar_loss=2.48.ckpt"
    CKPT_PATH = "./epoch=24-val_loss=3.48-val_insulin_loss=1.25-val_sugar_loss=2.23.ckpt"

    # params = {
    #     k: v
    #     for k, v in params.items()
    #     if k in inspect.signature(MixPredictMaskModel).parameters.keys()
    # }
    # father_model.model = MixPredictMaskModel(**params)
    # father_model.model.load_state_dict(torch.load(CKPT_PATH, map_location=torch.device("cpu")))
    # print(father_model.model.state_dict()["insulin_predict_dense.fc_2.bias"])
    # state_dict = torch.load(CKPT_PATH, map_location=torch.device("cpu"))["state_dict"]
    # new_state_dict = {}/
    # for k, v in state_dict.items():
    #     new_state_dict[k[6:]] = v
    # father_model.model.load_state_dict(new_state_dict)
    father_model.model.cuda()
    # print(father_model.model.state_dict()["insulin_predict_dense.fc_2.bias"])
    # assert 1==0
    # print(father_model.model.train_patient_list)  # pass
    # print(father_model.optimizer)  # pass
    # print(father_model.train_dataloader)  # pass

    # epoch_p_bar = tqdm(range(hp.epochs), total=hp.epochs)
    # for ep in epoch_p_bar:
    #     train_epoch_record = []
    #     val_epoch_record = []
    #     test_epoch_record = []
    #
    #     train_p_bar = tqdm(
    #         enumerate(father_model.train_dataloader), total=len(father_model.train_dataloader)
    #     )
    #     father_model.model.eval()
    #     for batch_index, batch_data in train_p_bar:
    #         sample_input = father_model.get_sample_input(batch_data, device='gpu')
    #         break
    #     break
    #         assert 1==0
    #         train_step_out_dict = model.training_step(batch_data, batch_index)
    #         train_epoch_record.append(train_step_out_dict)
    #         model.optimizer.zero_grad()
    #         train_step_out_dict['loss'].backward()
    #         model.optimizer.step()
    #         train_p_bar.set_description(f'Epoch {ep}/{hp.epochs} Train')
    #         train_p_bar.set_postfix(step_loss=train_step_out_dict['loss'].data.item())
    #     valid_p_bar = tqdm(enumerate(model.val_dataloader), total=len(model.val_dataloader))
    #     model.model.eval()
    #     for batch_index, batch_data in valid_p_bar:
    #         val_step_out_dict = model.validation_step(batch_data, batch_index)
    #         val_epoch_record.append(val_step_out_dict)
    #         valid_p_bar.set_description(f'Epoch {ep}/{hp.epochs} Valid')
    #         valid_p_bar.set_postfix(step_loss=val_step_out_dict['loss'].data.item())
    #     test_p_bar = tqdm(enumerate(model.test_dataloader), total=len(model.test_dataloader))
    #     model.model.eval()
    #     for batch_index, batch_data in test_p_bar:
    #         test_step_out_dict = model.test_step(batch_data, batch_index)
    #         test_epoch_record.append(test_step_out_dict)
    #         test_p_bar.set_description(f'Epoch {ep}/{hp.epochs} Test')
    #         test_p_bar.set_postfix(step_loss=test_step_out_dict['loss'].data.item())
    #
    #     train_epoch_out_dict = model.training_epoch_end(train_epoch_record)
    #     val_epoch_out_dict = model.validation_epoch_end(val_epoch_record)
    #     test_epoch_out_dict = model.test_epoch_end(test_epoch_record)
    #     model.lr_scheduler.step()
    #     epoch_p_bar.set_description(f'Epoch [{ep}/{hp.epochs}]')
    #     print('train_loss', train_epoch_out_dict['loss'].data.item(),
    #           'val_loss', val_epoch_out_dict['loss'].data.item(),
    #           'test_loss', test_epoch_out_dict['loss'].data.item(),)

    from datasets.utils import ExaminationEncoder
    exam_encoder = ExaminationEncoder('not_use')
    examination_list = exam_encoder.examination_list
    # shapleytaylor = GroupShapleyTaylor(father_model.model, sample_input)
    # shapleytaylor = GroupKernelShapleyTaylor(father_model.model, sample_input, examination_list=examination_list)

    # for each in father_model.train_dataset[0]:
    #     if torch.is_tensor(each):
    #         print(each.shape)
    #     else:
    #         print(type(each))
    # assert 1==0
    # print(father_model.test_dataset[0][0].keys())
    # test_batch_data = father_model.test_dataset[0][0]
    # father_model.model.cpu()
    # test_batch_data = father_model.get_sample_input(next(iter(father_model.test_dataloader)))
    # print(father_model.forward(test_batch_data))
    # assert 1==0


    PatientInfo.base_dir = PATIENT_BASE_DIR
    PatientInfo.util = DatasetEncodeUtil()

    # patient_id_list = [
    #     # "1566729",
    #     # "1392750",
    #     # "1397635",
    #     # "1032809",
    #     # "1581403",
    #     # "874732",
    #     # "1140742",
    #     # "1093903",
    #     # "451342",
    #     # "862006",
    #     # "1587847",
    #     # "1496860",
    #     # "986641",
    #     # "1519067",
    #     "1007192",
    # ]

    # load train dataset
    patient_id_list = []
    with open("./resources/train_filename_list.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            patient_id_list.append(line.split('.jso')[0])
    patient_list = [PatientInfo(f"{patient_id}.json") for patient_id in patient_id_list]
    train_dataset = MixPredictDataset(
        patient_list,
        max_feature_days=hp["max_feature_days"],
        min_feature_days=hp["min_feature_days"],
        sugar_miss_maximum=SUGAR_MISS_MAXIMUM,
    )
    print(f"There are {len(patient_list)} patients, {len(train_dataset)} pieces of data in the train set")
    # load test set
    patient_id_list = []
    with open("./resources/test_filename_list.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            patient_id_list.append(line.split('.jso')[0])
    patient_list = [PatientInfo(f"{patient_id}.json") for patient_id in patient_id_list]

    test_dataset = MixPredictDataset(
        patient_list,
        max_feature_days=hp["max_feature_days"],
        min_feature_days=hp["min_feature_days"],
        sugar_miss_maximum=SUGAR_MISS_MAXIMUM,
    )
    print(f"There are {len(patient_list)} patients, {len(test_dataset)} pieces of data in the train set")


    trainset_basal_predict = []
    trainset_premix_predict = []
    trainset_shot_predict = []

    trainset_basal_label = []
    trainset_premix_label = []
    trainset_shot_label = []
    # train set
    for i in tqdm(range(len(train_dataset))):
        x, (y_sugar, y_insulin) = train_dataset[i:i + 1]

        insulin_data = father_model.model.unpack_insulin_data(x)

        def tuple_to_gpu(_data):
            data = []
            for each in _data:
                if torch.is_tensor(each):
                    data.append(each.cuda())
                else:
                    data.append(each)
            return data
        insulin_output = father_model.model.insulin_forward(*tuple_to_gpu(insulin_data))

        # print(insulin_output.shape)  # torch.Size([1, 7])
        # print(y_insulin.shape)  # torch.Size([1, 7])

        for each_time in range(7):
            this_time_target_insulin_vec = x["insulin_vec_7_point"][0][2][each_time]  # 2 for the last (target) day
            # print(this_time_target_insulin_vec)  # tensor([10.0000,  1.0000,  0.5000,  0.5000,  2.0000, 12.0000, 14.0000, 24.0000, 12.0000])
            insulin_class_num = this_time_target_insulin_vec[1]
            if insulin_class_num == -1:
                continue
            elif insulin_class_num == 0:
                trainset_basal_predict.append(insulin_output[0, each_time].detach().cpu())
                trainset_basal_label.append(y_insulin[0, each_time])
            elif insulin_class_num == 1:
                trainset_premix_predict.append(insulin_output[0, each_time].detach().cpu())
                trainset_premix_label.append(y_insulin[0, each_time])
            elif insulin_class_num == 2:
                trainset_shot_predict.append(insulin_output[0, each_time].detach().cpu())
                trainset_shot_label.append(y_insulin[0, each_time])
            else:
                raise ValueError

    trainset_basal_label, trainset_basal_predict = torch.hstack(trainset_basal_label), torch.hstack(trainset_basal_predict)
    trainset_premix_label, trainset_premix_predict = torch.hstack(trainset_premix_label), torch.hstack(trainset_premix_predict)
    trainset_shot_label, trainset_shot_predict = torch.hstack(trainset_shot_label), torch.hstack(trainset_shot_predict)

    torch.save({
        "trainset_basal_label": trainset_basal_label,
        "trainset_basal_predict": trainset_basal_predict,
        "trainset_premix_label": trainset_premix_label,
        "trainset_premix_predict": trainset_premix_predict,
        "trainset_shot_label": trainset_shot_label,
        "trainset_shot_predict": trainset_shot_predict
    }, "yingzhen_trainset_predict_and_label.pth")

    testset_basal_predict = []
    testset_premix_predict = []
    testset_shot_predict = []

    testset_basal_label = []
    testset_premix_label = []
    testset_shot_label = []
    # test set
    for i in tqdm(range(len(test_dataset))):
        x, (y_sugar, y_insulin) = test_dataset[i:i + 1]

        # print(x["insulin_vec_7_point"])
        # print(x["insulin_vec_7_point"].shape)

        insulin_data = father_model.model.unpack_insulin_data(x)

        def tuple_to_gpu(_data):
            data = []
            for each in _data:
                if torch.is_tensor(each):
                    data.append(each.cuda())
                else:
                    data.append(each)
            return data

        insulin_output = father_model.model.insulin_forward(*tuple_to_gpu(insulin_data))

        # print(insulin_output.shape)  # torch.Size([1, 7])
        # print(y_insulin.shape)  # torch.Size([1, 7])

        for each_time in range(7):
            this_time_target_insulin_vec = x["insulin_vec_7_point"][0][2][each_time]  # 2 for the last (target) day
            # print(this_time_target_insulin_vec)  # tensor([10.0000,  1.0000,  0.5000,  0.5000,  2.0000, 12.0000, 14.0000, 24.0000, 12.0000])
            insulin_class_num = this_time_target_insulin_vec[1]
            if insulin_class_num == -1:
                continue
            elif insulin_class_num == 0:
                testset_basal_predict.append(insulin_output[0, each_time].detach().cpu())
                testset_basal_label.append(y_insulin[0, each_time])
            elif insulin_class_num == 1:
                testset_premix_predict.append(insulin_output[0, each_time].detach().cpu())
                testset_premix_label.append(y_insulin[0, each_time])
            elif insulin_class_num == 2:
                testset_shot_predict.append(insulin_output[0, each_time].detach().cpu())
                testset_shot_label.append(y_insulin[0, each_time])
            else:
                raise ValueError

    testset_basal_label, testset_basal_predict = torch.hstack(testset_basal_label), torch.hstack(testset_basal_predict)
    testset_premix_label, testset_premix_predict = torch.hstack(testset_premix_label), torch.hstack(testset_premix_predict)
    testset_shot_label, testset_shot_predict = torch.hstack(testset_shot_label), torch.hstack(testset_shot_predict)

    torch.save({
        "testset_basal_label": testset_basal_label,
        "testset_basal_predict": testset_basal_predict,
        "testset_premix_label": testset_premix_label,
        "testset_premix_predict": testset_premix_predict,
        "testset_shot_label": testset_shot_label,
        "testset_shot_predict": testset_shot_predict
    }, "yingzhen_testset_predict_and_label.pth")


    assert 1==0


    for each_test_sample_index in tqdm(random_index[hp["patient_limit_start"]:hp["patient_limit_end"]]):
        # each_test_sample_index = 2628
    # for each_test_sample_index in tqdm([2775]):
        # for each_test_sample_index in tqdm([2775, 2778, 2780]):
        output_txt_path = f"./{hp['save_dir']}/translate{each_test_sample_index}.txt"

        x, (y_sugar, y_insulin) = test_dataset[each_test_sample_index:each_test_sample_index+1]
        print("Success", x["patient_id"], x["patient_day"], each_test_sample_index)
        # build yesterday data
        yesterday_dataset = MixPredictDataset_targetpidday(
            [PatientInfo(f"{int(x['patient_id'][0].numpy())}.json")],
            max_feature_days=hp["max_feature_days"],
            min_feature_days=hp["min_feature_days"],
            sugar_miss_maximum=SUGAR_MISS_MAXIMUM,
            target_patient_day=int(x["patient_day"])-1
        )
        x_yesterday, (y_sugar_yesterday, y_insulin_yesterday) = yesterday_dataset[0:1]
        print(x["examination_vec"])
        print(x_yesterday["examination_vec"])
        print(x["sugar_vec_7_point"])
        assert x["examination_vec"].sum() == x_yesterday["examination_vec"].sum()
        assert x["sugar_vec_7_point"][0, 1].sum() == x_yesterday["sugar_vec_7_point"][0, 2].sum()

        # if x["patient_id"] == 1004801:
        #     print(each_test_sample_index)
        # continue
        output_txt = open(output_txt_path, "w")
        print("Patient ID", int(x["patient_id"].data.item()), file=output_txt)
        output_txt.flush()
        # print(x.keys())  # dict_keys(['patient_id', 'examination_vec', 'day_vec_7_point', 'day_vec_24_point', 'point_vec_7_point', 'point_vec_24_point', 'sugar_vec_7_point', 'sugar_vec_24_point', 'drug_vec_1_point', 'drug_vec_7_point', 'drug_vec_24_point', 'insulin_vec_4_point', 'insulin_vec_7_point', 'insulin_vec_24_point', 'insulin_vec_7_point_temp', 'insulin_vec_24_point_temp'])

        insulin_data = father_model.model.unpack_insulin_data(x)
        sugar_data = father_model.model.unpack_sugar_data(x)

        insulin_data_yesterday = father_model.model.unpack_insulin_data(x_yesterday)
        sugar_data_yesterday = father_model.model.unpack_sugar_data(x_yesterday)

        def tuple_to_gpu(_data):
            data= []
            for each in _data:
                if torch.is_tensor(each):
                    data.append(each.cuda())
                else:
                    data.append(each)
            return data

        insulin_output = father_model.model.insulin_forward(*tuple_to_gpu(insulin_data))
        sugar_output = father_model.model.sugar_forward(*tuple_to_gpu(sugar_data))
        print('Success', insulin_output, sugar_output)

        test_batch_data = (sugar_data, insulin_data, y_sugar, y_insulin)
        test_batch_data_yesterday = (sugar_data_yesterday, insulin_data_yesterday, y_sugar, y_insulin)

        np.save(f"./{hp['save_dir']}/sample_input_tuple_yesterday{each_test_sample_index}.npy", insulin_data_yesterday)

        shapleytaylor = GroupKernelShapleyTaylor(father_model.model, test_batch_data,
                                                 examination_list=examination_list,
                                                 patient_limit=each_test_sample_index,
                                                 output_txt=output_txt,
                                                 save_dir=hp["save_dir"])
        # test_sugar_data, test_insulin_data = test_batch_data
        example_batch_data = shapley_utils.build_all_missing_baseline(test_batch_data)
        example_sugar_data, example_insulin_data = example_batch_data
        # shapleytaylor.assign_baseline_data(test_insulin_data)
        # return_info = shapleytaylor.assign_baseline_data(example_insulin_data)

        # print(type(example_insulin_data))
        # for each in example_insulin_data:
        #     print(type(each))
        # print(type(test_batch_data_yesterday))
        # for each in test_batch_data_yesterday:
        #     print(type(each))

        return_info = shapleytaylor.assign_baseline_data(list(insulin_data_yesterday))

        if return_info == "No target insulin, continue":
            os.system(f"rm -rf ./{hp['save_dir']}/translate{each_test_sample_index}.txt")
            os.system(f"rm -rf ./{hp['save_dir']}/sample_input_tuple{each_test_sample_index}.npy")
            continue
        # shap_values = shapleytaylor.calculate_shapley_taylor(max_k=2)
        shap_values = shapleytaylor.calculate_shapley_taylor(max_k=2, use_narrow=True)
        # shap_values = shapleytaylor.calculate_shapley_taylor_stack(max_k=2)

        # assert 1 == 0


def get_argument_parser():

    parser = ArgumentParser()

    parser.add_argument(
        "--base_dir", type=str, default="/mnt/sda1/libiao/models/sugar/mix/"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=1)

    parser.add_argument("--ckpt_name", type=str, default="model")
    parser.add_argument("--loss_name", type=str, default="rmse")

    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--lr_scheduler_name", type=str, default="cosine")

    parser.add_argument("--max_feature_days", type=int, default=3)
    parser.add_argument("--min_feature_days", type=int, default=3)

    parser.add_argument("--patient_limit", type=int, default=20000)
    parser.add_argument("--patient_limit_start", type=int, default=0)
    parser.add_argument("--patient_limit_end", type=int, default=0)
    parser.add_argument("--sugar_start_index", type=int, default=0)
    parser.add_argument("--sugar_end_index", type=int, default=7)
    parser.add_argument("--sugar_miss_maximum", type=int, default=4)

    parser.add_argument("--mlp_encode_dilation", type=int, default=5)
    parser.add_argument("--mlp_output_dim", type=int, default=256)
    parser.add_argument("--mlp_encode_blocks", type=int, default=5)
    parser.add_argument("--mlp_encode_dropout", type=float, default=0)

    parser.add_argument("--dense_hidden_size", type=int, default=256)
    parser.add_argument("--dense_dropout", type=float, default=0.3)

    parser.add_argument("--pool_name", type=str, default="max")
    parser.add_argument("--activation_name", type=str, default="smu")

    parser.add_argument("--attn_num_heads", type=int, default=8)
    parser.add_argument("--attn_dropout", type=float, default=0)

    parser.add_argument("--rnn_name", type=str, default="gru")
    parser.add_argument("--rnn_hidden_size", type=int, default=512)
    parser.add_argument("--rnn_num_layers", type=int, default=2)

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--shapley_force", action="store_true")

    parser.add_argument("--save_dir", type=str, default="assess_save_tmp")

    args = parser.parse_known_args()[0].__dict__
    return args


if __name__ == "__main__":
    print('start')
    args = get_argument_parser()
    main(args)
