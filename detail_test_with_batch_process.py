import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np

trainset_stat = torch.load("yingzhen_trainset_predict_and_label_with_batch.pth")
testset_stat = torch.load("yingzhen_testset_predict_and_label_with_batch.pth")

trainset_basal_label = trainset_stat["trainset_basal_label"]
trainset_basal_predict = trainset_stat["trainset_basal_predict"]
trainset_premix_label = trainset_stat["trainset_premix_label"]
trainset_premix_predict = trainset_stat["trainset_premix_predict"]
trainset_shot_label = trainset_stat["trainset_shot_label"]
trainset_shot_predict = trainset_stat["trainset_shot_predict"]
yesterday_trainset_basal_label = trainset_stat["yesterday_trainset_basal_label"]
yesterday_trainset_basal_predict = trainset_stat["yesterday_trainset_basal_predict"]
yesterday_trainset_premix_label = trainset_stat["yesterday_trainset_premix_label"]
yesterday_trainset_premix_predict = trainset_stat["yesterday_trainset_premix_predict"]
yesterday_trainset_shot_label = trainset_stat["yesterday_trainset_shot_label"]
yesterday_trainset_shot_predict = trainset_stat["yesterday_trainset_shot_predict"]


testset_basal_label = testset_stat["testset_basal_label"]
testset_basal_predict = testset_stat["testset_basal_predict"]
testset_premix_label = testset_stat["testset_premix_label"]
testset_premix_predict = testset_stat["testset_premix_predict"]
testset_shot_label = testset_stat["testset_shot_label"]
testset_shot_predict = testset_stat["testset_shot_predict"]
yesterday_testset_basal_label = testset_stat["yesterday_testset_basal_label"]
yesterday_testset_basal_predict = testset_stat["yesterday_testset_basal_predict"]
yesterday_testset_premix_label = testset_stat["yesterday_testset_premix_label"]
yesterday_testset_premix_predict = testset_stat["yesterday_testset_premix_predict"]
yesterday_testset_shot_label = testset_stat["yesterday_testset_shot_label"]
yesterday_testset_shot_predict = testset_stat["yesterday_testset_shot_predict"]

total_train_label = torch.concat((trainset_basal_label, trainset_premix_label, trainset_shot_label))
total_train_predict = torch.concat((trainset_basal_predict, trainset_premix_predict, trainset_shot_predict))

total_test_label = torch.concat((testset_basal_label, testset_premix_label, testset_shot_label))
total_test_predict = torch.concat((testset_basal_predict, testset_premix_predict, testset_shot_predict))


# basal
print("Basal")
print("Number:", trainset_basal_label.shape[0])
print("Train MAE:", torch.mean(torch.abs(trainset_basal_label - trainset_basal_predict)))
print("Train RMSE:", torch.sqrt(torch.mean((trainset_basal_label - trainset_basal_predict)**2)))
print("Train R2:", r2_score(trainset_basal_label, trainset_basal_predict))
print("Train Pearson:", pearsonr(trainset_basal_label, trainset_basal_predict))
print("Number:", testset_basal_label.shape[0])
print("Test MAE:", torch.mean(torch.abs(testset_basal_label - testset_basal_predict)))
print("Test RMSE:", torch.sqrt(torch.mean((testset_basal_label - testset_basal_predict)**2)))
print("Test R2:", r2_score(testset_basal_label, testset_basal_predict))
print("Test Pearson:", pearsonr(testset_basal_label, testset_basal_predict))

# premix
print("Premix")
print("Number:", trainset_premix_label.shape[0])
print("Train MAE:", torch.mean(torch.abs(trainset_premix_label - trainset_premix_predict)))
print("Train RMSE:", torch.sqrt(torch.mean((trainset_premix_label - trainset_premix_predict)**2)))
print("Train R2:", r2_score(trainset_premix_label, trainset_premix_predict))
print("Train Pearson:", pearsonr(trainset_premix_label, trainset_premix_predict))
print("Number:", testset_premix_label.shape[0])
print("Test MAE:", torch.mean(torch.abs(testset_premix_label - testset_premix_predict)))
print("Test RMSE:", torch.sqrt(torch.mean((testset_premix_label - testset_premix_predict)**2)))
print("Test R2:", r2_score(testset_premix_label, testset_premix_predict))
print("Test Pearson:", pearsonr(testset_premix_label, testset_premix_predict))

# shot
print("Shot")
print("Number:", trainset_shot_label.shape[0])
print("Train MAE:", torch.mean(torch.abs(trainset_shot_label - trainset_shot_predict)))
print("Train RMSE:", torch.sqrt(torch.mean((trainset_shot_label - trainset_shot_predict)**2)))
print("Train R2:", r2_score(trainset_shot_label, trainset_shot_predict))
print("Train Pearson:", pearsonr(trainset_shot_label, trainset_shot_predict))
print("Number:", testset_shot_label.shape[0])
print("Test MAE:", torch.mean(torch.abs(testset_shot_label - testset_shot_predict)))
print("Test RMSE:", torch.sqrt(torch.mean((testset_shot_label - testset_shot_predict)**2)))
print("Test R2:", r2_score(testset_shot_label, testset_shot_predict))
print("Test Pearson:", pearsonr(testset_shot_label, testset_shot_predict))

# total
print("Total")
print("Number:", total_train_label.shape[0])
print("Train MAE:", torch.mean(torch.abs(total_train_label - total_train_predict)))
print("Train RMSE:", torch.sqrt(torch.mean((total_train_label - total_train_predict)**2)))
print("Train R2:", r2_score(total_train_label, total_train_predict))
print("Train Pearson:", pearsonr(total_train_label, total_train_predict))
print("Number:", total_test_label.shape[0])
print("Test MAE:", torch.mean(torch.abs(total_test_label - total_test_predict)))
print("Test RMSE:", torch.sqrt(torch.mean((total_test_label - total_test_predict)**2)))
print("Test R2:", r2_score(total_test_label, total_test_predict))
print("Test Pearson:", pearsonr(total_test_label, total_test_predict))


# yesterday compare
# basal
trainset_basal_yesterday_label_copare = trainset_basal_label - yesterday_trainset_basal_label
testset_basal_yesterday_label_copare = testset_basal_label - yesterday_testset_basal_label
trainset_basal_yesterday_predict_copare = trainset_basal_predict - yesterday_trainset_basal_predict
testset_basal_yesterday_predict_copare = testset_basal_predict - yesterday_testset_basal_predict

# premix
trainset_premix_yesterday_label_copare = trainset_premix_label - yesterday_trainset_premix_label
testset_premix_yesterday_label_copare = testset_premix_label - yesterday_testset_premix_label
trainset_premix_yesterday_predict_copare = trainset_premix_predict - yesterday_trainset_premix_predict
testset_premix_yesterday_predict_copare = testset_premix_predict - yesterday_testset_premix_predict


# shot
trainset_shot_yesterday_label_copare = trainset_shot_label - yesterday_trainset_shot_label
testset_shot_yesterday_label_copare = testset_shot_label - yesterday_testset_shot_label
trainset_shot_yesterday_predict_copare = trainset_shot_predict - yesterday_trainset_shot_predict
testset_shot_yesterday_predict_copare = testset_shot_predict - yesterday_testset_shot_predict

yesterday_label_compare = torch.concat([trainset_basal_yesterday_label_copare,
                                                 testset_basal_yesterday_label_copare,
                                                 trainset_premix_yesterday_label_copare,
                                                 testset_premix_yesterday_label_copare,
                                                 trainset_shot_yesterday_label_copare,
                                                 testset_shot_yesterday_label_copare])
yesterday_predict_compare = torch.concat([trainset_basal_yesterday_predict_copare,
                                                 testset_basal_yesterday_predict_copare,
                                                 trainset_premix_yesterday_predict_copare,
                                                 testset_premix_yesterday_predict_copare,
                                                 trainset_shot_yesterday_predict_copare,
                                                 testset_shot_yesterday_predict_copare])

# label doctor
print( (yesterday_predict_compare[torch.where(yesterday_label_compare > 0)] > 0).sum() / len(yesterday_label_compare) )
print( (yesterday_predict_compare[torch.where(yesterday_label_compare < 0)] < 0).sum() / len(yesterday_label_compare) )
