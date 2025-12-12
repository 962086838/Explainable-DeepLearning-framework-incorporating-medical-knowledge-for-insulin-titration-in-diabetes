import torch
from torch import Tensor

from utils.constants import *


def stepped_mean_absolutely_ratio_error(predict: Tensor, label: Tensor) -> Tensor:
    mask_1 = (label > 0) & (label < 3.9)
    mask_2 = (label >= 3.9) & (label < 7.8)
    mask_3 = label >= 7.8
    loss = 0
    if mask_1.sum():
        loss += (
            torch.mean(torch.abs(predict[mask_1] - label[mask_1]) / label[mask_1])
        ) * 3
    if mask_2.sum():
        loss += (
            torch.mean(torch.abs(predict[mask_2] - label[mask_2]) / label[mask_2])
        ) * 1
    if mask_3.sum():
        loss += (
            torch.mean(torch.abs(predict[mask_3] - label[mask_3]) / label[mask_3])
        ) * 2
    # r1 = (torch.nanmean(torch.abs(predict[mask_1] - label[mask_1]) / label[mask_1]))
    # r2 = (torch.nanmean(torch.abs(predict[mask_2] - label[mask_2]) / label[mask_2]))
    # r3 = (torch.nanmean(torch.abs(predict[mask_3] - label[mask_3]) / label[mask_3]))
    # loss = r1 * 3 + r2 + r3 * 2
    return loss


def mean_absolutely_error(predict: Tensor, label: Tensor) -> Tensor:
    mask = label > 0
    diff = torch.abs(predict[mask] - label[mask])
    loss = torch.mean(diff)
    return loss


def rmse_loss(predict: Tensor, label: Tensor) -> Tensor:
    mask = label > 0
    diff = predict[mask] - label[mask]
    loss = torch.sqrt(torch.mean(diff**2))
    return loss


def mean_absolutely_ratio_error(predict: Tensor, label: Tensor) -> Tensor:
    predict = predict[label > 0]
    label = label[label > 0]
    loss = torch.mean(torch.abs((predict - label) / label))
    return loss


def rmse_ratio_loss(predict: Tensor, label: Tensor) -> Tensor:
    predict = predict[label > 0]
    label = label[label > 0]

    diff = (predict - label) ** 2
    ratio = torch.sqrt(torch.mean(diff / label))
    return ratio


def part_rmse_loss(diff: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(diff**2))


def weighted_nan_to_sum_with_sugar(hypoglycemia, hyperglycemia, normal):
    return (
        torch.nan_to_num(hypoglycemia) * 1
        + torch.nan_to_num(hyperglycemia) * 2
        + torch.nan_to_num(normal) * 3
    )


def weighted_rmse_loss_2(predict: Tensor, label: Tensor) -> Tensor:
    pass


def weighted_rmse_loss(predict: Tensor, label: Tensor) -> Tensor:
    label_before_meal = label[:, [0, 2, 4]]
    predict_before_meal = predict[:, [0, 2, 4]]

    label_after_meal = label[:, [1, 3, 5, 6]]
    predict_after_meal = predict[:, [1, 3, 5, 6]]

    # 饭前，label有值
    label_before_meal_base_mask = label_before_meal > 0
    # 饭前低血糖
    label_before_meal_hypoglycemia_mask = label_before_meal_base_mask & (
        label_before_meal <= BEFORE_MEAL_SUGAR_NORMAL_RANGE[0]
    )
    # 饭前高血糖
    label_before_meal_hyperglycemia_mask = label_before_meal_base_mask & (
        label_before_meal >= BEFORE_MEAL_SUGAR_NORMAL_RANGE[1]
    )
    # 饭前正常血糖
    label_before_meal_normal_mask = (
        ~(label_before_meal_hypoglycemia_mask | label_before_meal_hyperglycemia_mask)
        & label_before_meal_base_mask
    )

    # 饭后，label有值
    label_after_meal_base_mask = label_after_meal > 0
    # 饭前低血糖
    label_after_meal_hypoglycemia_mask = label_after_meal_base_mask & (
        label_after_meal <= AFTER_MEAL_SUGAR_NORMAL_RANGE[0]
    )
    # 饭前高血糖
    label_after_meal_hyperglycemia_mask = label_after_meal_base_mask & (
        label_after_meal >= AFTER_MEAL_SUGAR_NORMAL_RANGE[1]
    )
    # 饭前正常血糖
    label_after_meal_normal_mask = (
        ~(label_after_meal_hypoglycemia_mask | label_after_meal_hyperglycemia_mask)
        & label_after_meal_base_mask
    )

    diff_before_meal = label_before_meal - predict_before_meal
    diff_after_meal = label_after_meal - predict_after_meal

    loss_before_meal = weighted_nan_to_sum_with_sugar(
        part_rmse_loss(diff_before_meal[label_before_meal_hypoglycemia_mask]),
        part_rmse_loss(diff_before_meal[label_before_meal_hyperglycemia_mask]),
        part_rmse_loss(diff_before_meal[label_before_meal_normal_mask]) * 1,
    )

    loss_after_meal = weighted_nan_to_sum_with_sugar(
        part_rmse_loss(diff_after_meal[label_after_meal_hypoglycemia_mask]),
        part_rmse_loss(diff_after_meal[label_after_meal_hyperglycemia_mask]),
        part_rmse_loss(diff_after_meal[label_after_meal_normal_mask]),
    )

    loss = (loss_before_meal + loss_after_meal) / 2
    return loss
