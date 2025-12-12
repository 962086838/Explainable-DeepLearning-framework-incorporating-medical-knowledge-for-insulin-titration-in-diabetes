PATIENT_BASE_DIR = "../DataPreprocess/data/patients_all/version001/"

TRAIN_PATIENT_LIST = "./resources/train_filename_list.txt"
VAL_PATIENT_LIST = "./resources/test_filename_list.txt"
TEST_PATIENT_LIST = "./resources/test_filename_list.txt"

# 餐前血糖范围【严格、一般、宽松】
BEFORE_MEAL_SUGAR_STRICT_RANGE = (4.4, 6.1)
BEFORE_MEAL_SUGAR_NORMAL_RANGE = (6.1, 7.8)
BEFORE_MEAL_SUGAR_EASING_RANGE = (7.8, 10.0)
# 餐后血糖范围【严格、一般、宽松】
AFTER_MEAL_SUGAR_STRICT_RANGE = (6.1, 7.8)
AFTER_MEAL_SUGAR_NORMAL_RANGE = (7.8, 10.0)
AFTER_MEAL_SUGAR_EASING_RANGE = (7.8, 13.9)

# 缺失值
DEFAULT_MISSING_VALUE = -1

DAY_DIM = 1
POINT_DIM = 1
SUGAR_DIM = 1
INSULIN_DIM = 9
DRUG_DIM = 28