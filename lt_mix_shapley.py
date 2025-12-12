import os
import random
import inspect
from typing import Any, Dict
import numpy as np
import pytorch_lightning as lt
from torch import Tensor
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from datasets.patient import load_patient_list
from models.mix_predict_mask import MixPredictMaskModel
from datasets.dataset.mix_predict import MixPredictDataset, MixPredictDataset_targetpidday
import shapley_utils
from lt_mix_nni import set_seed, MixPredictModule

from utils.constants import PATIENT_BASE_DIR
from datasets.patient import PatientInfo
from datasets.utils import DatasetEncodeUtil
import utils
from utils.utils import judge_plan_type, judge_plan_type_time
'''
CUDA_VISIBLE_DEVICES=0 python lt_mix_shapley.py --gpus=1 --epochs=800 --ckpt_name="w10_mae_mix_model" --loss_name=mae --optimizer_name=adam --lr_scheduler_name=cosine --batch_size=1 --patient_limit_start 0 --patient_limit_end 500 --save_dir assess_save_20230101_all 
'''


from tqdm import tqdm

from shapley_taylor import ShapleyTaylor, GroupShapleyTaylor, GroupKernelShapleyTaylor, GroupKernelShapleyTaylor_onlysugarposneg

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
    CKPT_PATH = "./epoch=24-val_loss=3.48-val_insulin_loss=1.25-val_sugar_loss=2.23.ckpt"
    father_model = MixPredictModule.load_from_checkpoint(CKPT_PATH, hp=hp)

    father_model.eval()


    father_model.model.cuda()

    from datasets.utils import ExaminationEncoder
    exam_encoder = ExaminationEncoder('not_use')
    examination_list = exam_encoder.examination_list


    PatientInfo.base_dir = PATIENT_BASE_DIR
    PatientInfo.util = DatasetEncodeUtil()

    patient_id_list = []
    with open("./resources/cleaned_test_filename_list.txt", "r") as f:
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

    test_dataset_length = len(test_dataset)
    random_index = np.arange(test_dataset_length).tolist()


    # first generate treatment plan query list
    candidate_random_dict = {}
    for each_test_sample_index in tqdm(random_index):
        x, (y_sugar, y_insulin) = test_dataset[each_test_sample_index:each_test_sample_index + 1]
        _plan_type = judge_plan_type_time(x["insulin_vec_7_point"])
        if _plan_type in candidate_random_dict.keys():
            candidate_random_dict[_plan_type].append(each_test_sample_index)
        else:
            candidate_random_dict[_plan_type] = [each_test_sample_index]




    for each_test_sample_index in tqdm(random_index[hp["patient_limit_start"]:min(hp["patient_limit_end"], 6356)]):

        output_txt_path = f"./{hp['save_dir']}/translate{each_test_sample_index}.txt"


        x, (y_sugar, y_insulin) = test_dataset[each_test_sample_index:each_test_sample_index+1]
        print("Success", x["patient_id"], x["patient_day"], each_test_sample_index)
        # build yesterday data
        yesterday_dataset = MixPredictDataset_targetpidday(
            [PatientInfo(f"{int(patient_id)}.json") for patient_id in x['patient_id'].numpy()],
            max_feature_days=hp["max_feature_days"],
            min_feature_days=hp["min_feature_days"],
            sugar_miss_maximum=SUGAR_MISS_MAXIMUM,
            target_patient_day=(x["patient_day"].numpy()).astype(int)-1
        )
        x_yesterday, (y_sugar_yesterday, y_insulin_yesterday) = yesterday_dataset[0:1]

        print("==============Warning today data loaded==============")

        _plan_type = judge_plan_type_time(x["insulin_vec_7_point"])
        fake_translate_success_flag = "FailEmptyOverlap"

        if len(_plan_type)==0:
            os.system(f"rm -rf ./{hp['save_dir']}/translate{each_test_sample_index}.txt")
            # os.system(f"rm -rf ./{hp['save_dir']}/translate{each_test_sample_index}_fake{random_patient_index}.txt")
            os.system(f"rm -rf ./{hp['save_dir']}/sample_input_tuple{each_test_sample_index}.npy")
            os.system(f"rm -rf ./{hp['save_dir']}/sample_input_tuple_yesterday{each_test_sample_index}.npy")
            continue
        else:
            while fake_translate_success_flag != "Success":  # startswith
                random_patient_index = random.choice(candidate_random_dict[_plan_type])
                x_random, (y_sugar_random, y_insulin_random) = test_dataset[random_patient_index:random_patient_index + 1]

                output_txt_origin_path = f"./{hp['save_dir']}/translate{each_test_sample_index}_origin.txt"
                output_txt_onlysugarposneg_path = f"./{hp['save_dir']}/translate{each_test_sample_index}_onlysugarposneg.txt"
                output_fake_txt_path = f"./{hp['save_dir']}/translate{each_test_sample_index}_fake{random_patient_index}.txt"
                random_patient_yesterday_dataset = MixPredictDataset_targetpidday(
                    [PatientInfo(f"{int(patient_id)}.json") for patient_id in x_random['patient_id'].numpy()],
                    max_feature_days=hp["max_feature_days"],
                    min_feature_days=hp["min_feature_days"],
                    sugar_miss_maximum=SUGAR_MISS_MAXIMUM,
                    target_patient_day=(x_random["patient_day"].numpy()).astype(int)-1
                )
                x_random_yesterday, (y_sugar_random_yesterday, y_insulin_random_yesterday) = random_patient_yesterday_dataset[0:1]


                print(x["examination_vec"])
                print(x_yesterday["examination_vec"])
                print(x["sugar_vec_7_point"])
                assert x["examination_vec"].sum() == x_yesterday["examination_vec"].sum()
                assert x["sugar_vec_7_point"][0, 1].sum() == x_yesterday["sugar_vec_7_point"][0, 2].sum()

                output_txt = open(output_txt_path, "w")
                output_origin_txt = open(output_txt_origin_path, "w")
                output_onlysugarposneg_txt = open(output_txt_onlysugarposneg_path, "w")
                output_fake_txt = open(output_fake_txt_path, "w")
                print("Patient ID", int(x["patient_id"].data.item()), file=output_txt)
                print("Patient ID", int(x["patient_id"].data.item()), file=output_origin_txt)
                print("Patient ID", int(x["patient_id"].data.item()), file=output_onlysugarposneg_txt)
                print("Patient ID", int(x["patient_id"].data.item()), file=output_fake_txt)
                output_txt.flush()
                output_origin_txt.flush()
                output_onlysugarposneg_txt.flush()
                output_fake_txt.flush()

                insulin_data = father_model.model.unpack_insulin_data(x)
                sugar_data = father_model.model.unpack_sugar_data(x)
                insulin_data_yesterday = father_model.model.unpack_insulin_data(x_yesterday)
                sugar_data_yesterday = father_model.model.unpack_sugar_data(x_yesterday)

                insulin_data_random = father_model.model.unpack_insulin_data(x_random)
                sugar_data_random = father_model.model.unpack_sugar_data(x_random)
                insulin_data_random_yesterday = father_model.model.unpack_insulin_data(x_random_yesterday)
                sugar_data_random_yesterday = father_model.model.unpack_sugar_data(x_random_yesterday)

                # 昨天数据用全部missing代替，需要搭配下面的assign_baseline_data(list(insulin_data_yesterday), ignore_yesterday_missing=False)
                ignore_yesterday_missing = False  # default False
                ignore_yesterday_exam_missing = True  # default False
                print(type(insulin_data_yesterday))
                insulin_data_yesterday_setmissing = []
                for _i, each_i in enumerate(insulin_data_yesterday):  # exam, insulin, temp insulin, sugar, drug, time, (a list)
                    if _i in [0]:  # exam
                        if ignore_yesterday_missing or ignore_yesterday_exam_missing:
                            insulin_data_yesterday_setmissing.append(torch.ones_like(each_i) * -1)
                    elif _i in [1, 2, 3]:  # insulin, temp insulin, sugar
                        if ignore_yesterday_missing:
                            insulin_data_yesterday_setmissing.append(torch.ones_like(each_i) * -1)
                        elif ignore_yesterday_exam_missing:
                            insulin_data_yesterday_setmissing.append(each_i)
                    elif _i == 4:  # drug
                        if ignore_yesterday_missing:
                            insulin_data_yesterday_setmissing.append(torch.zeros_like(each_i))
                        elif ignore_yesterday_exam_missing:
                            insulin_data_yesterday_setmissing.append(each_i)
                    elif _i in [5, 6]:  # time
                        if ignore_yesterday_missing:
                            insulin_data_yesterday_setmissing.append(each_i)
                        elif ignore_yesterday_exam_missing:
                            insulin_data_yesterday_setmissing.append(each_i)
                    else:
                        raise ValueError
                insulin_data_yesterday = insulin_data_yesterday_setmissing


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
                test_batch_data_random = (sugar_data_random, insulin_data_random, y_sugar_random, y_insulin_random)
                # test_batch_data_yesterday = (sugar_data_yesterday, insulin_data_yesterday, y_sugar, y_insulin)

                np.save(f"./{hp['save_dir']}/sample_input_tuple_yesterday{each_test_sample_index}.npy", insulin_data_yesterday)

                shapleytaylor = GroupKernelShapleyTaylor(father_model.model, test_batch_data,
                                                         examination_list=examination_list,
                                                         patient_limit=each_test_sample_index,
                                                         output_txt=output_txt,
                                                         save_dir=hp["save_dir"])


                shapleytaylor_random = GroupKernelShapleyTaylor(father_model.model, test_batch_data_random,
                                                                examination_list=examination_list,
                                                                patient_limit=each_test_sample_index,
                                                                output_txt=output_fake_txt,
                                                                save_dir=None)

                example_batch_data = shapley_utils.build_all_missing_baseline(test_batch_data)
                example_sugar_data, example_insulin_data = example_batch_data


                import utils.translate as translate

                model, full_coeff, model_baseline_prediction, model_prediction, _predictesidual, \
                target_insulin_time_list, y_insulin, f_translate, super_index_to_name, super_index_to_index, \
                sample_insulin_data_flatten, normal_index_to_narrow_index, final_epoch_output_info, narrow_index_to_normal_index = shapleytaylor.calculate_shapley_taylor(max_k=2, use_narrow=True)
                translate.translate(model, full_coeff, model_baseline_prediction, model_prediction, _predictesidual,
                      target_insulin_time_list, y_insulin, f_translate, super_index_to_name, super_index_to_index,
                      sample_insulin_data_flatten, normal_index_to_narrow_index, final_epoch_output_info)


                model_r, full_coeff_r, model_baseline_prediction_r, model_prediction_r, _predict_residual_r, \
                target_insulin_time_list_r, y_insulin_r, f_translate_r, super_index_to_name_r, super_index_to_index_r, \
                sample_insulin_data_flatten_r, normal_index_to_narrow_index_r, final_epoch_output_info_r, narrow_index_to_normal_index_r = shapleytaylor_random.calculate_shapley_taylor(max_k=2, use_narrow=True)



                print(len(model.translation_sentence_list_full), len(model_r.translation_sentence_list_full))
                overlap_translate_normal_index = []
                for _i, (_sentence, _sentence_r) in enumerate(zip(model.translation_sentence_list_full, model_r.translation_sentence_list_full)):
                    if len(_sentence) > 0 and len(_sentence_r) > 0:  # both have valid sentence
                        overlap_translate_normal_index.append(_i)

                print("overlap_translate_normal_index", overlap_translate_normal_index)

                print("Warning try to generate fake translation")
                fake_translate_success_flag = translate.translate_fake_random_sentence(model.translation_sentence_list_full,
                                                                                       model_r.translation_sentence_list_full,
                                                                                       model,
                                                                                       full_coeff, full_coeff_r,
                                                                                       f_translate_r,
                                                                                       model_baseline_prediction,
                                                                                       model_prediction,
                                                                                       _predictesidual,
                                                                                       target_insulin_time_list,
                                                                                       y_insulin,
                                                                                       final_epoch_output_info,
                                                                                       super_index_to_name,
                                                                                       normal_index_to_narrow_index,
                                                                                       sample_insulin_data_flatten,
                                                                                       super_index_to_index)

                if fake_translate_success_flag.startswith("Fail"):
                    os.system(f"rm -rf ./{hp['save_dir']}/translate{each_test_sample_index}_fake{random_patient_index}.txt")




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
    parser.add_argument("--patient_specific_id", type=int, default=None)
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
