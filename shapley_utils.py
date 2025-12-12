# Python3 implementation to find nCr

from math import *
import numpy as np
from collections import defaultdict
import torch
# Function to find the nCr
from scipy.optimize import lsq_linear
import time
from torch.autograd import Variable
from torch.optim import SGD, Adam, Adagrad, RMSprop
import re
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DECREASE_SUGAR_DRUG_MASK = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
INCREASE_SUGAR_DRUG_MASK = [22, 23, 24, 25, 26, 27]
_DECREASE_SUGAR_DRUG_MASK = np.zeros(28)
_DECREASE_SUGAR_DRUG_MASK[np.array(DECREASE_SUGAR_DRUG_MASK)] = 1
_INCREASE_SUGAR_DRUG_MASK = np.zeros(28)
_INCREASE_SUGAR_DRUG_MASK[np.array(INCREASE_SUGAR_DRUG_MASK)] = 1

_SUGAR_NORMAL_RANGE_LOW_BEFOREMEAL = 3.9
_SUGAR_NORMAL_RANGE_HIGH_BEFOREMEAL = 6.1
_SUGAR_NORMAL_RANGE_LOW_AFTERMEAL = 6.1
_SUGAR_NORMAL_RANGE_HIGH_AFTERMEAL = 8

TIME_NAME = ['早餐前', '早餐后', '午餐前', '午餐后', '晚餐前', '晚餐后', '睡前']
insulin_detail = json.load(open('./resources/insulin_detail.json'))['insulin_list']
# drug_detail = json.load(open('./resources/drug_detail.json'))['drug_list']  # 错误的顺序
drug_detail = ["伏格列波糖片", "利拉鲁肽注射液", "利格列汀片", "格列吡嗪控释片", "格列吡嗪片", "格列喹酮片", "格列美脲片", "格列齐特片(II)", "格列齐特缓释片", "沙格列汀片", "注射用甲泼尼龙琥珀酸钠", "瑞格列奈片", "甲泼尼龙片", "盐酸二甲双胍片", "盐酸二甲双胍缓释片", "盐酸吡格列酮片", "磷酸西格列汀片", "维格列汀片", "艾塞那肽注射液", "苯甲酸阿格列汀片", "西格列汀二甲双胍片", "那格列奈片", "醋酸可的松片", "醋酸地塞米松片", "醋酸泼尼松片", "醋酸泼尼松龙片", "阿卡波糖片", "马来酸罗格列酮片"]
INSULIN_DETAIL_NAME_LIST = [each['name'] for each in insulin_detail]
# DRUG_DETAIL_NAME_LIST = [each['name'] for each in drug_detail]
DRUG_DETAIL_NAME_LIST = drug_detail

JIBENXINXI = "体重、舒张压、身高、age、gender"
HUAYAN_YIDAOGONGNENG = "胰岛功能：120min血糖、180min血糖、240min血糖、2min 血糖、300min血糖、30min血糖、4min 血糖、60min血糖、6min 血糖、C肽、C肽 120分钟、C肽 180分钟、C肽 240分钟、C肽 2分钟、C肽 300分钟、C肽 30分钟、C肽 4分钟、C肽 60分钟、C肽 6分钟 C肽 空腹、C肽1、胰岛素、胰岛素 120分钟 胰岛素 180分钟 胰岛素 240分钟 胰岛素 2分钟 胰岛素 300分钟 胰岛素 30分钟 胰岛素 4分钟 胰岛素 60分钟 胰岛素 6分钟 胰岛素 空腹 餐后2h血糖 餐后血糖 空腹血糖 糖化白蛋白 糖化血红蛋白 葡萄糖 葡萄糖（急）"
HUAYAN_XUESHENGHUA = "β-羟丁酸、丙氨酸氨基转移酶、丙草比、乳酸、估算肾小球滤过率(根据CKD-EPI方程) 低密度脂蛋白33、低密度脂蛋白胆固醇1、氨基末端利钠肽前体、游离脂肪酸、甘油三酯1、直/总比、直接胆红素、总胆固醇1、总胆红素、总胆红素（急）结合胆红素（急） 肌酐 胆红素 酮体 门冬氨酸氨基转移酶 门冬氨酸氨基转移酶(急) "
HUAYAN_YIDAOSUZISHENKANGTI = "抗胰岛素自身抗体 抗胰岛细胞抗体 抗谷氨酸脱羧酶抗体"
HUAYAN_JISU = "胰岛素样生长因子 胰高血糖素 皮质醇 皮质醇1"
HUAYAN_NIAOSHENGHUA = "24小时尿白蛋白、24小时尿糖、24小时尿肌酐、尿液葡萄糖定量、尿白蛋白、尿白蛋白/肌肝"

IMPORTANT_EXAMINATION = ['体重', 'age', 'C肽', '胰岛素 空腹', '糖化血红蛋白', '估算肾小球滤过率(根据CKD-EPI方程)', '肌酐', '酮体', '皮质醇', '皮质醇1']

def printNcR(n, r):
    # p holds the value of n*(n-1)*(n-2)...,
    # k holds the value of r*(r-1)...
    p = 1
    k = 1

    # C(n, r) == C(n, n-r),
    # choosing the smaller value
    if (n - r < r):
        r = n - r

    if (r != 0):
        while (r):
            p *= n
            k *= r

            # gcd of p, k
            m = gcd(p, k)

            # dividing by gcd, to simplify product
            # division by their gcd saves from
            # the overflow
            p //= m
            k //= m

            n -= 1
            r -= 1

        # k should be simplified to 1
        # as C(n, r) is a natural number
        # (denominator should be 1 )

    else:
        p = 1

    # if our approach is correct p = ans and k =1
    # print(p)
    return p


def C(n, m, method='dp'):
    """
    从n个不同元素中选取m个元素的组合数
    C(n, m) = n! / (m! * (n - m)!)
    :param n: int,总元素个数
    :param m: int,需要选择的元素个数
    :param method: str,下述列表字符其中之一:['dc', 'dp', 'log', 'prime']
                   - dc:暴力相除 - dp:动态规划 - log:log求解法，非精确解 - prime:质因数分解法
    :return: res 组合数
    """
    if method == 'dc':
        # 暴力相除，有溢出风险，python最大float约为1.79e+308
        # 时间复杂度 O(n), 空间复杂度O(1)
        nums1, nums2 = 1, 1
        m = min(m, n - m)
        for x in range(1, m + 1):
            nums1 *= x
            nums2 *= x + n - m
        return nums2 // nums1
    elif method == 'dp':
        # DP，较慢
        # 时间复杂度 O(nm) 空间复杂度 O(mn) 可以优化到O(m)
        # 从n个元素中取m个元素可以划分为两个子问题：
        # 对于元素i,选它等于从n-1个元素中选取m-1个元素
        # 对于元素i，不选它等于从n-1个元素中选取m个元素
        # dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
        # 初始化：从i个元素中选取0个元素的组合数为1，从0个元素中选取j(>0)个元素的组合数为0
        # 遍历顺序：从上往下从左往右遍历即可
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(m + 1, i + 1)):
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1]
        return dp[-1][-1]
    elif method == 'log':
        # log估算法,不够精确
        # 时间复杂度 O(n) 空间复杂度 O(1)
        log_value = 0
        m = min(m, n - m)
        for i in range(1, m + 1):
            log_value = log_value + np.log(n - m + i) - np.log(i)
        return int(round(np.exp(log_value), 0))
    elif method == 'prime':
        # 质数分解法
        # 时间复杂度 O(n ** 2) 空间复杂度 O(n)  看上去很高，但实际运行中质数越往后越稀疏，比DP快很多
        # C的值必为非负整数，分母可以分解为若干质数，分子也可以分解为若干质数，将等量的质数相除，得到纯分母
        # 这里的相除不需要真实相除，只需要计算每个质数的个数然后将个数相减即可
        primes = set()
        for i in range(2, n + 1):
            is_prime = True
            for j in range(2, int(np.sqrt(i)) + 1):
                if i % j == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.add(i)

        n_factors = defaultdict(int)
        m_factors = defaultdict(int)
        m = min(m, n - m)

        for i in range(1, m + 1):
            k = i
            while k not in primes and k > 1:
                for p in primes:
                    if k % p == 0:
                        k = k // p
                        m_factors[p] += 1
                        break
            if k > 1:
                m_factors[k] += 1

            l = n - m + i
            while l not in primes and l > 1:
                for p in primes:
                    if l % p == 0:
                        l = l // p
                        n_factors[p] += 1
                        break
            if l > 1:
                n_factors[l] += 1

        res = 1
        for k, v in n_factors.items():
            res *= k ** (v - m_factors[k])
        return int(res)


class TorchRidge:
    def __init__(self, alpha=0, fit_intercept=True, ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.tensor, y: torch.tensor, sample_weight=None) -> None:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(torch.float32)
        if isinstance(sample_weight, np.ndarray):
            sample_weight = torch.from_numpy(sample_weight).to(torch.float32)

        X = X.cuda()
        y = y.cuda()
        sample_weight = sample_weight.cuda()

        X = X.rename(None)
        y = y.rename(None).squeeze().unsqueeze(1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        # print(X.shape)  # torch.Size([1000, 16836])
        # print(y.shape)  # torch.Size([1000, 1, 7])
        if len(y.shape)==3 and y.shape[1]==1:
            y = y.squeeze(dim=1)
        assert (len(y.shape) == 2)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1).to(X.device), X], dim=1)

        if sample_weight is not None:
            # print(sample_weight.shape)  # torch.Size([600])
            # print(torch.sqrt(sample_weight).shape)  # torch.Size([600])
            W2 = torch.sqrt(sample_weight).squeeze().unsqueeze(1)
            # print(W2.shape)  # torch.Size([600, 1])
            y = y * W2
            X = X * W2

        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y
        # print(X.shape)
        torch.save(X, 'tmp_X.pth')
        torch.save(y, 'tmp_y.pth')
        lhs = X.T @ X
        rhs = X.T @ y
        del X, y, sample_weight
        if self.alpha == 0:
            self.w = torch.linalg.lstsq(lhs, rhs).solution
            torch.save(lhs, 'tmp_lhs.pth')
            torch.save(rhs, 'tmp_rhs.pth')
            mse_f = torch.nn.MSELoss()
            print('Warning alpha == 0 mse_f of residual', mse_f(torch.matmul(lhs, self.w), rhs))
            torch.save(self.w, 'tmp_w.pth')
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0]).to(torch.float64)
            ridge = ridge.to(lhs.device)
            torch.save(lhs + ridge, 'tmp_lhs.pth')
            torch.save(rhs, 'tmp_rhs.pth')
            self.w = torch.linalg.lstsq(lhs + ridge, rhs).solution  # torch.Size([301, 7])
            mse_f = torch.nn.MSELoss()
            print('Warning alpha != 0 mse_f of residual', mse_f(torch.matmul(lhs + ridge, self.w), rhs))
            torch.save(self.w, 'tmp_w.pth')
        if self.fit_intercept:
            self.intercept_ = self.w[0].detach().cpu()
            self.coef_ = self.w[1:]
        else:
            self.coef_ = self.w
        self.coef_ = self.coef_.t().detach().cpu()
        # self.fit_residual_matrix = torch.matmul(lhs + ridge, self.w) - rhs
        print(self.coef_.shape)  # torch.Size([7, 16836])

    def predict(self, X: torch.tensor) -> None:
        # X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1).to(X.device), X], dim=1)
        # print(X, self.w)
        return X @ self.w.to(torch.float)


class Constrained_least_square:
    def __init__(self, target_insulin_time_list, narrow_bound, alpha=0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.target_insulin_time_list = target_insulin_time_list
        self.narrow_bound = narrow_bound
        self.lb_expansion, self.ub_expansion = self.narrow_bound_expansion()

    def expansion_f_lb(self, _a, _b):
        result = []
        for _i, a in enumerate(_a):
            for b in _b[_i:]:
                if a==-np.inf and b==-np.inf:
                    result.append(-np.inf)
                elif (a == -np.inf and b == 0) or (a == 0 and b == -np.inf ):
                    result.append(-np.inf)
                elif a == 0 and b == 0:
                    result.append(0)
                else:
                    print(a, b)
                    raise ValueError
        return np.array(result)

    def expansion_f_ub(self, _a, _b):
        result = []
        for _i, a in enumerate(_a):
            for b in _b[_i:]:
                if a==np.inf and b==np.inf:
                    result.append(np.inf)
                elif (a == np.inf and b == 0) or (a == 0 and b == np.inf ):
                    result.append(np.inf)
                elif a == 0 and b == 0:
                    result.append(0)
                else:
                    print(a, b)
                    raise ValueError
        return np.array(result)

    def narrow_bound_expansion(self):
        lb_list = []
        ub_list = []
        for i in range(len(self.narrow_bound)):
            lb_list.append(self.narrow_bound[i][0])
            ub_list.append(self.narrow_bound[i][1])
        lb_list = np.array(lb_list)
        ub_list = np.array(ub_list)
        lb_expansion = np.apply_along_axis(lambda s: self.expansion_f_lb(s, s), 0, lb_list)  # (300,)
        ub_expansion = np.apply_along_axis(lambda s: self.expansion_f_ub(s, s), 0, ub_list)  # (300,)
        print(lb_list.shape)
        print('lb_expansion.shape', lb_expansion.shape)
        print('ub_expansion.shape', ub_expansion.shape)
        return lb_expansion, ub_expansion

    def fit(self, X: torch.tensor, y: torch.tensor, sample_weight=None) -> None:
        # only support numpy
        # if isinstance(X, np.ndarray):
        #     X = torch.from_numpy(X).to(torch.float32)
        # if isinstance(y, np.ndarray):
        #     y = torch.from_numpy(y).to(torch.float32)
        if isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.numpy()
        # X = X.rename(None)
        # y = y.rename(None).squeeze().unsqueeze(1)
        print(X.shape, y.shape)  # (600, 300) (600, 7)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if len(y.shape)==3 and y.shape[1]==1:
            y = y.squeeze(dim=1)
        assert (len(y.shape) == 2)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])  # (600, 301)
            self.lb_expansion = np.hstack([-np.inf, self.lb_expansion])  # (301,)
            self.ub_expansion = np.hstack([np.inf, self.ub_expansion])  # (301,)
            # print('lb_expansion.shape', self.lb_expansion.shape)
            # print('ub_expansion.shape', self.ub_expansion.shape)
        if sample_weight is not None:
            W2 = np.sqrt(sample_weight).reshape(-1, 1)
            y = y * W2
            X = X * W2

        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y
        # print(X.shape)
        lhs = X.T @ X
        rhs = X.T @ y
        print(lhs.shape)  # (301, 301)
        print(rhs.shape)  # (301, 7)
        # del X, y, sample_weight
        if self.alpha == 0:
            result_list = []
            for _each_index in self.target_insulin_time_list:
                res = lsq_linear(lhs, rhs[:, _each_index], bounds=(-np.inf, np.inf), lsmr_tol='auto', verbose=1)  # 不加限制
                # res = lsq_linear(lhs, rhs[:, _each_index], bounds=(self.lb_expansion, self.ub_expansion), lsmr_tol='auto', verbose=1)  # 加限制
                result_list.append(res.x.reshape(-1, 1))
        else:
            raise NotImplementedError
        self.w = np.hstack(result_list)
        if self.fit_intercept:
            self.intercept_ = self.w[0]
            self.coef_ = self.w[1:]
        else:
            self.coef_ = self.w
        self.coef_ = torch.Tensor(self.coef_.T).cuda()
        # self.fit_residual_matrix = torch.matmul(lhs + ridge, self.w) - rhs
        print('self.coef_.shape', self.coef_.shape)  # torch.Size([4, 300])

    def predict(self, X) -> None:
        # X = X.rename(None)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        # print(X, self.w)
        return torch.FloatTensor(X @ self.w).cuda()


def judge_drug_data_increase_or_decrease_sugar(drug_data):
    _increase_dot = np.dot(_INCREASE_SUGAR_DRUG_MASK, drug_data)
    _decrease_dot = np.dot(_DECREASE_SUGAR_DRUG_MASK, drug_data)
    if np.sum(_increase_dot) > 0 and np.sum(_decrease_dot) == 0:
        return 'increase'
    elif np.sum(_increase_dot) == 0 and np.sum(_decrease_dot) > 0:
        return 'decrease'
    else:
        return 'hybrid'


def judge_sugar_data_range(sugar_data, time):
    if sugar_data > 0:
        # print(sugar_data)
        if time in [0, 2, 4, 6]:  # 餐前 + 睡前
            if sugar_data < _SUGAR_NORMAL_RANGE_LOW_BEFOREMEAL:
                return 'low'
            elif sugar_data > _SUGAR_NORMAL_RANGE_HIGH_BEFOREMEAL:
                return 'high'
            else:
                return 'mid'
        elif time in [1, 3, 5]:  # 餐后
            if sugar_data < _SUGAR_NORMAL_RANGE_LOW_AFTERMEAL:
                return 'low'
            elif sugar_data > _SUGAR_NORMAL_RANGE_HIGH_AFTERMEAL:
                return 'high'
            else:
                return 'mid'
        else:
            raise NotImplementedError
    else:
        return 'mid'


class GradientDescendRidge:  # abs weight ratio, pos neg weight
    """
    self.sample_data_baseline_data_common_dim_difference

    """
    def __init__(self, sample_insulin_data, baseline_data, sample_insulin_data_flatten_narrow, baseline_data_flatten_narrow,
                 w_l2=0, w_l1=0,  fit_intercept=True, lr=0.01, init_method='analytic',
                 narrow_index_to_name=None,
                 super_index_to_name=None,
                 super_index_narrowed_baselineonly=None,
                 super_index_narrowed_dataonly=None,
                 narrow_index_to_normal_index=None,
                 insulin_l1=0, insulin_l2=0, exam_l1=0, exam_l2=0,
                 sugar_l1=0,
                 sugar_pos_neg=True,
                 drug_pos_neg=True,
                 mid_sugar_insulin_limit=True,
                 sugar_high_range_insulin_pos=True,
                 target_insulin_time_list=None,
                 target_insulin_type_list=None,
                 ):
        self.target_insulin_time_list = target_insulin_time_list
        self.target_insulin_type_list = target_insulin_type_list
        self.build_target_insulin_related_sugar_time_list()
        self.sample_insulin_data = sample_insulin_data
        self.baseline_data = baseline_data
        self.sample_insulin_data_flatten_narrow = sample_insulin_data_flatten_narrow
        self.baseline_data_flatten_narrow = baseline_data_flatten_narrow
        self.narrow_index_to_normal_index = narrow_index_to_normal_index
        self.normal_index_to_narrow_index = {value: key for key, value in self.narrow_index_to_normal_index.items()}
        self.w_l2 = w_l2
        self.w_l1 = w_l1
        self.insulin_l1 = insulin_l1
        self.insulin_l2 = insulin_l2
        self.exam_l2 = exam_l2
        self.exam_l1 = exam_l1
        self.sugar_l1 = sugar_l1
        self.sugar_pos_neg = sugar_pos_neg
        self.drug_pos_neg = drug_pos_neg
        self.mid_sugar_insulin_limit = mid_sugar_insulin_limit
        self.sugar_high_range_insulin_pos = sugar_high_range_insulin_pos
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.mse_loss = torch.nn.MSELoss()
        self.init_method = init_method
        self.narrow_index_to_name = narrow_index_to_name
        self.super_index_to_name = super_index_to_name
        self.super_index_narrowedornot_baselineonly = super_index_narrowed_baselineonly
        self.super_index_narrowedornot_dataonly = super_index_narrowed_dataonly
        self.sugar_value = self.sample_insulin_data[3][torch.where(self.sample_insulin_data[3]!=-1)]
        self.sugar_value_yesterday = self.baseline_data[3][torch.where(self.sample_insulin_data[3]!=-1)]

        assert self.init_method in ['random', 'analytic']

        self._exam_index_offset = 0
        self._insulin_index_offset = int(self._exam_index_offset + self.sample_insulin_data[0].shape[1])
        self._temp_insulin_index_offset = int(self._insulin_index_offset + self.sample_insulin_data[1].shape[1] / 9)
        self._sugar_insulin_index_offset = int(self._temp_insulin_index_offset + self.sample_insulin_data[2].shape[1] / 9)
        self._drug_index_offset = int(self._sugar_insulin_index_offset + self.sample_insulin_data[3].shape[1] / 1)
        print("offset information",
              self._exam_index_offset,
              self._insulin_index_offset,
              self._temp_insulin_index_offset,
              self._sugar_insulin_index_offset,
              self._drug_index_offset)

        self.matrix_insulin_interaction_index = None
        self.matrix_sugar_interaction_index = None
        self.matrix_exam_interaction_index = None
        self.build_interaction_index()
        self.build_first_order_second_order_matrix()
        # build data和baseline相交部分的pos neg，最终得到self._sample_data_baseline_data_intersect_minus_pos_neg
        # data和baseline只有各自有的部分的pos neg，self._sample_data_baseline_data_nooverlap_pos_neg
        self.build_data_baseline_true_available_matrix()
        self.build_pos_neg_basedontrueavailablematrix()
        # build translation sentence with self._sample_data_baseline_data_intersect_minus_pos_neg 和 self._sample_data_baseline_data_nooverlap_pos_neg
        self.build_translation_sentence()   # 产生self.translation_sentence_list，长度是baseline和data的并集
        self.build_time_related_index()
        self.build_global_pos_neg_index()
        self.build_insulin_sugar_interaction_time_related_pos_neg_matrix()



    def return_drug_data_by_drug_index(self, _i):
        # drug data & sugar 都可以用
        this_name = self.narrow_index_to_name[_i]
        day, time = int(re.findall('\d+', this_name)[0]), int(re.findall('\d+', this_name)[1])

        if 'drug' in this_name:
            drug_type = int(re.findall('\d+', this_name)[2])
            return self.sample_insulin_data[4][0, int(day * 7 + time) * 28 + drug_type]
        elif 'sugar' in this_name:
            return self.sample_insulin_data[3][0, int(day * 7 + time) * 1: int(day * 7 + time + 1) * 1]
        else:
            raise NotImplementedError

    def build_first_order_second_order_matrix(self):
        self.matrix_first_order_index = np.eye(len(self.narrow_index_to_name))
        self.matrix_second_order_index = np.triu(np.ones((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))) - self.matrix_first_order_index

        self.matrix_first_order_index = np.triu(self.matrix_first_order_index)
        self.matrix_first_order_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_first_order_index)
        self.matrix_second_order_index = np.triu(self.matrix_second_order_index)
        self.matrix_second_order_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_second_order_index)

    def build_insulin_sugar_interaction_time_related_pos_neg_matrix(self):  # 考虑胰岛素、血糖时间离得近的（作用时间内影响）的正负
        self.insulin_sugar_interaction_time_related_pos_neg_matrix = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in range(self.insulin_sugar_interaction_time_related_pos_neg_matrix.shape[0]):
            for _j in range(_i, self.insulin_sugar_interaction_time_related_pos_neg_matrix.shape[1]):  # 只处理上三角
                _i_name = self.narrow_index_to_name[_i]  # must be insulin
                _j_name = self.narrow_index_to_name[_j]  # must be sugar

                if 'insulin' in _i_name:
                    _baseline_insulin_rawdata = self._baseline_data_value_slice_list[_i].cpu().numpy()
                    _sample_insulin_rawdata = self._sample_data_value_slice_list[_i].cpu().numpy()
                    if _baseline_insulin_rawdata[0]==_sample_insulin_rawdata[0] and _baseline_insulin_rawdata[-1]!=_sample_insulin_rawdata[-1]:  # 必须是同类胰岛素，baseline和sample之间的剂量不同才考虑
                        if 'sugar' in _j_name:
                            _baseline_sugar_rawdata = self._baseline_data_value_slice_list[_j].cpu().numpy()
                            _sample_sugar_rawdata = self._sample_data_value_slice_list[_j].cpu().numpy()
                            _sample_baseline_sugar_diff = _sample_sugar_rawdata - _baseline_sugar_rawdata
                            if _baseline_sugar_rawdata != -1 and _sample_sugar_rawdata != -1:  # baseline和sample的 sugar 都有值
                                _sample_insulin_name_index = int(_sample_insulin_rawdata[0])
                                insulin_day_time = re.findall("\d+", _i_name)
                                insulin_day, insulin_time = int(insulin_day_time[0]), int(insulin_day_time[1])
                                sugar_day_time = re.findall("\d+", _j_name)
                                sugar_day, sugar_time = int(sugar_day_time[0]), int(sugar_day_time[1])
                                _insulin_type = insulin_detail[_sample_insulin_name_index]["classes"][0]  # shot basal premix
                                _sample_insulin_dose_value = _sample_insulin_rawdata[-1]
                                _baseline_insulin_dose_value = _sample_insulin_rawdata[-1]
                                _sample_baseline_insulin_dose_diff = _sample_insulin_dose_value - _baseline_insulin_dose_value
                                if _insulin_type == 'shot':  # 作用时间到当餐餐后
                                    if 0 < (sugar_day * 7 + sugar_time - insulin_day * 7 - insulin_time) <= 1:
                                        if _sample_baseline_insulin_dose_diff>0 and _sample_baseline_sugar_diff>0:  # 判断胰岛素剂量和血糖上升/下降
                                            self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                        elif _sample_baseline_insulin_dose_diff<0 and _sample_baseline_sugar_diff>0:
                                            _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata, sugar_time)
                                            if _high_mid_low == 'high':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            if _high_mid_low == 'low':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = -1
                                        elif _sample_baseline_insulin_dose_diff>0 and _sample_baseline_sugar_diff<0:  # 	胰岛素↑+血糖↓：
                                            _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata, sugar_time)
                                            if _high_mid_low=='high':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            if _high_mid_low=='low':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = -1
                                        else:  # 胰岛素↓+血糖↓
                                            _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata, sugar_time)
                                            if _high_mid_low=='high':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            if _high_mid_low=='low':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = -1
                                elif _insulin_type == 'premix':  # 作用时间到当餐餐后->下一餐餐后
                                    if 0 < (sugar_day * 7 + sugar_time - insulin_day * 7 - insulin_time) <= 3:
                                        if _sample_baseline_insulin_dose_diff > 0 and _sample_baseline_sugar_diff > 0:  # 判断胰岛素剂量和血糖上升/下降
                                            self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                        elif _sample_baseline_insulin_dose_diff < 0 and _sample_baseline_sugar_diff > 0:
                                            _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata, sugar_time)
                                            if _high_mid_low == 'high':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            if _high_mid_low == 'low':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = -1
                                        elif _sample_baseline_insulin_dose_diff > 0 and _sample_baseline_sugar_diff < 0:  # 胰岛素↑+血糖↓：
                                            _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata, sugar_time)
                                            if _high_mid_low == 'high':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            if _high_mid_low == 'low':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = -1
                                        else:  # 胰岛素↓+血糖↓
                                            _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata, sugar_time)
                                            if _high_mid_low == 'high':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            if _high_mid_low == 'low':
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = -1
                                elif _insulin_type == 'basal':  # 作用时间早上注射和晚上注射不一样。
                                    if insulin_time == 0:
                                        if 0 <= (sugar_day * 7 + sugar_time - insulin_day * 7 - insulin_time) <= 6:
                                            if _sample_baseline_insulin_dose_diff > 0 and _sample_baseline_sugar_diff > 0:  # 判断胰岛素剂量和血糖上升/下降
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            elif _sample_baseline_insulin_dose_diff < 0 and _sample_baseline_sugar_diff > 0:
                                                _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata,
                                                                                       sugar_time)
                                                if _high_mid_low == 'high':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = 1
                                                if _high_mid_low == 'low':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = -1
                                            elif _sample_baseline_insulin_dose_diff > 0 and _sample_baseline_sugar_diff < 0:  # 胰岛素↑+血糖↓：
                                                _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata,
                                                                                       sugar_time)
                                                if _high_mid_low == 'high':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = 1
                                                if _high_mid_low == 'low':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = -1
                                            else:  # 胰岛素↓+血糖↓
                                                _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata,
                                                                                       sugar_time)
                                                if _high_mid_low == 'high':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = 1
                                                if _high_mid_low == 'low':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = -1
                                    if insulin_time == 6:
                                        if 0 < (sugar_day * 7 + sugar_time - insulin_day * 7 - insulin_time) <= 1:
                                            if _sample_baseline_insulin_dose_diff > 0 and _sample_baseline_sugar_diff > 0:  # 判断胰岛素剂量和血糖上升/下降
                                                self.insulin_sugar_interaction_time_related_pos_neg_matrix[_i, _j] = 1
                                            elif _sample_baseline_insulin_dose_diff < 0 and _sample_baseline_sugar_diff > 0:
                                                _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata,
                                                                                       sugar_time)
                                                if _high_mid_low == 'high':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = 1
                                                if _high_mid_low == 'low':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = -1
                                            elif _sample_baseline_insulin_dose_diff > 0 and _sample_baseline_sugar_diff < 0:  # 胰岛素↑+血糖↓：
                                                _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata,
                                                                                       sugar_time)
                                                if _high_mid_low == 'high':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = 1
                                                if _high_mid_low == 'low':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = -1
                                            else:  # 胰岛素↓+血糖↓
                                                _high_mid_low = judge_sugar_data_range(_sample_sugar_rawdata,
                                                                                       sugar_time)
                                                if _high_mid_low == 'high':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = 1
                                                if _high_mid_low == 'low':
                                                    self.insulin_sugar_interaction_time_related_pos_neg_matrix[
                                                        _i, _j] = -1
                                else: raise ValueError
        self.insulin_sugar_interaction_time_related_pos_neg_matrix = np.triu(self.insulin_sugar_interaction_time_related_pos_neg_matrix)
        self.insulin_sugar_interaction_time_related_pos_neg_matrix = self.strech_upper_tri_matrix_to_flatten(
            self.insulin_sugar_interaction_time_related_pos_neg_matrix)


    def build_interaction_index(self):
        # build index information lists
        _exam_index = []
        _exam_important_index = []
        self._insulin_index = []
        self._temp_insulin_index = []
        _sugar_index = []
        _drug_index = []

        _sugar_low_range_index = []
        _sugar_mid_range_index = []
        _sugar_high_range_index = []
        _sugar_before_meal_index = []
        _sugar_after_meal_index = []
        self._drug_increase_sugar_index = []
        self._drug_decrease_sugar_index = []

        for _i in range(len(self.narrow_index_to_name)):
            this_name = self.narrow_index_to_name[_i]

            # 与target时间无关的内容
            if 'temp_insulin_day' in this_name:
                self._temp_insulin_index.append(_i)
            elif 'insulin_day' in this_name:
                self._insulin_index.append(_i)
            elif 'sugar_day' in this_name:
                day_time = re.findall("\d+", this_name)  # 输出结果为列表
                day = int(day_time[0])
                time = int(day_time[1])
                _sugar_index.append(_i)
                # 对于血糖，进一步看看具体数值
                _sugar_data = self.return_drug_data_by_drug_index(_i)
                _sugar_effect = judge_sugar_data_range(_sugar_data, time)
                eval(f'_sugar_{_sugar_effect}_range_index').append(_i) #  _sugar_low_range_index _sugar_mid_range_index _sugar_high_range_index
                # 对于血糖，进一步看看是餐前还是餐后
                if time in [0, 2, 4]:
                    _sugar_before_meal_index.append(_i)
                elif time in [1, 3, 5]:
                    _sugar_after_meal_index.append(_i)
                else:
                    assert time==6
            elif 'drug_day' in this_name:
                day_time = re.findall("\d+", this_name)  # 输出结果为列表
                _drug_type = int(day_time[2])
                _drug_index.append(_i)
                # 对于药物，进一步看看是否全部是降低/升高血糖的药物
                _drug_data = self.return_drug_data_by_drug_index(_i)
                # _drug_effect = judge_drug_data_increase_or_decrease_sugar(_drug_data)
                if _drug_type in INCREASE_SUGAR_DRUG_MASK:
                    self._drug_increase_sugar_index.append(_i)
                elif _drug_type in DECREASE_SUGAR_DRUG_MASK:
                    self._drug_decrease_sugar_index.append(_i)
                else:
                    raise ValueError
            else:
                assert not this_name.startswith('day_')  # 检查条目通过
                _exam_index.append(_i)
                if self.narrow_index_to_name[_i] in IMPORTANT_EXAMINATION:
                    _exam_important_index.append(_i)

        # build temp insulin 临时胰岛素 interaction index
        self.matrix_temp_insulin_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._temp_insulin_index:
            for _j in self._temp_insulin_index:
                self.matrix_temp_insulin_interaction_index[_i, _j] = 1
        self.matrix_temp_insulin_interaction_index = np.triu(self.matrix_temp_insulin_interaction_index)
        self.matrix_temp_insulin_interaction_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_temp_insulin_interaction_index)
        # build 胰岛素 interaction index
        self.matrix_insulin_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._insulin_index:
            for _j in self._insulin_index:
                self.matrix_insulin_interaction_index[_i, _j] = 1
        self.matrix_insulin_interaction_index = np.triu(self.matrix_insulin_interaction_index)
        self.matrix_insulin_interaction_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_insulin_interaction_index)
        # build 胰岛素 血糖 interaction index
        self.matrix_insulin_sugar_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._insulin_index:
            for _j in _sugar_index:
                self.matrix_insulin_sugar_interaction_index[_i, _j] = 1
        self.matrix_insulin_sugar_interaction_index = np.triu(self.matrix_insulin_sugar_interaction_index)
        self.matrix_insulin_sugar_interaction_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_insulin_sugar_interaction_index)
        # build 血糖 interaction index
        self.matrix_sugar_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in _sugar_index:
            for _j in _sugar_index:
                self.matrix_sugar_interaction_index[_i, _j] = 1
        self.matrix_sugar_interaction_index = np.triu(self.matrix_sugar_interaction_index)
        self.matrix_sugar_interaction_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_sugar_interaction_index)
        # build 血糖 > 高的边 interaction index
        self.matrix_sugar_interaction_pos_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in range(len(_sugar_index)):
            for _j in range(len(_sugar_index)):
                if _sugar_index[_i] in _sugar_high_range_index and _sugar_index[_j] in _sugar_high_range_index:
                    self.matrix_sugar_interaction_pos_index[_sugar_index[_i], _sugar_index[_j]] = 1
        self.matrix_sugar_interaction_pos_index = np.triu(self.matrix_sugar_interaction_pos_index)
        self.matrix_sugar_interaction_pos_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_sugar_interaction_pos_index)
        # build 血糖 < 低的边 interaction index
        self.matrix_sugar_interaction_neg_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in range(len(_sugar_index)):
            for _j in range(len(_sugar_index)):
                if _sugar_index[_i] in _sugar_low_range_index and _sugar_index[_j] in _sugar_low_range_index:
                    self.matrix_sugar_interaction_neg_index[_sugar_index[_i], _sugar_index[_j]] = 1
        self.matrix_sugar_interaction_neg_index = np.triu(self.matrix_sugar_interaction_neg_index)
        self.matrix_sugar_interaction_neg_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_sugar_interaction_neg_index)
        # build 血糖 > _SUGAR_NORMAL_RANGE_HIGH & insulin
        self.matrix_sugar_high_range_insulin_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._insulin_index:
            for _j in range(len(_sugar_index)):
                assert _sugar_index[_j] in _sugar_low_range_index + _sugar_mid_range_index + _sugar_high_range_index
                if _sugar_index[_j] in _sugar_high_range_index:
                    self.matrix_sugar_high_range_insulin_interaction_index[_i, _sugar_index[_j]] = 1
                # if self.sugar_value[_j] > _SUGAR_NORMAL_RANGE_HIGH:
                #     self.matrix_sugar_high_range_insulin_interaction_index[_i, _sugar_index[_j]] = 1
        self.matrix_sugar_high_range_insulin_interaction_index = np.triu(self.matrix_sugar_high_range_insulin_interaction_index)
        self.matrix_sugar_high_range_insulin_interaction_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_sugar_high_range_insulin_interaction_index)
        # build sugar mid range & insulin interaction index
        self.matrix_sugar_mid_range_insulin_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._insulin_index:
            for _j in _sugar_mid_range_index:
                self.matrix_sugar_mid_range_insulin_interaction_index[_i, _j] = 1
        self.matrix_sugar_mid_range_insulin_interaction_index = np.triu(self.matrix_sugar_mid_range_insulin_interaction_index)
        self.matrix_sugar_mid_range_insulin_interaction_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_sugar_mid_range_insulin_interaction_index)
        # build 血糖 after meal & insulin
        self.matrix_sugar_after_meal_insulin_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._insulin_index:
            for _j in _sugar_after_meal_index:
                self.matrix_sugar_after_meal_insulin_interaction_index[_i, _j] = 1
        self.matrix_sugar_after_meal_insulin_interaction_index = np.triu(self.matrix_sugar_after_meal_insulin_interaction_index)
        self.matrix_sugar_after_meal_insulin_interaction_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_sugar_after_meal_insulin_interaction_index)
        # build exam interaction index
        self.matrix_exam_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in _exam_index:
            for _j in _exam_index:
                self.matrix_exam_interaction_index[_i, _j] = 1
        self.matrix_exam_interaction_index = np.triu(self.matrix_exam_interaction_index)
        self.matrix_exam_interaction_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_exam_interaction_index)
        # build exam other interaction index
        self.matrix_exam_other_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in _exam_index:
            for _j in self._insulin_index+self._temp_insulin_index+_sugar_index+_drug_index:
                self.matrix_exam_other_interaction_index[_i, _j] = 1
        self.matrix_exam_other_interaction_index = np.triu(self.matrix_exam_other_interaction_index)
        self.matrix_exam_other_interaction_index = self.strech_upper_tri_matrix_to_flatten(
            self.matrix_exam_other_interaction_index)
        # build exam important exam interaction index
        self.matrix_exam_important_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in _exam_important_index:
            for _j in _exam_important_index:
                self.matrix_exam_important_interaction_index[_i, _j] = 1
        self.matrix_exam_important_interaction_index = np.triu(self.matrix_exam_important_interaction_index)
        self.matrix_exam_important_interaction_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_exam_important_interaction_index)
        # build drug interaction index
        self.matrix_drug_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in _drug_index:
            for _j in _drug_index:
                self.matrix_drug_interaction_index[_i, _j] = 1
        self.matrix_drug_interaction_index = np.triu(self.matrix_drug_interaction_index)
        self.matrix_drug_interaction_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_drug_interaction_index)
        # build drug (increase sugar) interaction index
        self.matrix_drug_increase_sugar_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._drug_increase_sugar_index:
            for _j in self._drug_increase_sugar_index:
                self.matrix_drug_increase_sugar_interaction_index[_i, _j] = 1
        self.matrix_drug_increase_sugar_interaction_index = np.triu(self.matrix_drug_increase_sugar_interaction_index)
        self.matrix_drug_increase_sugar_interaction_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_drug_increase_sugar_interaction_index)
        # build drug (decrease sugar) interaction index
        self.matrix_drug_decrease_sugar_interaction_index = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in self._drug_decrease_sugar_index:
            for _j in self._drug_decrease_sugar_index:
                self.matrix_drug_decrease_sugar_interaction_index[_i, _j] = 1
        self.matrix_drug_decrease_sugar_interaction_index = np.triu(self.matrix_drug_decrease_sugar_interaction_index)
        self.matrix_drug_decrease_sugar_interaction_index = self.strech_upper_tri_matrix_to_flatten(self.matrix_drug_decrease_sugar_interaction_index)

    def build_data_baseline_true_available_matrix(self):
        '''
        self.super_index_narrowed = {}  # 维度是否被narrow了，key为未压缩的数量，value为True/False,  下面两个的的OR运算
        self.super_index_narrowed_dataonly = {}  # 维度是否被narrow了，key为未压缩的数量，value为True/False
        self.super_index_narrowed_baselineonly = {}  # 维度是否被narrow了，key为未压缩的数量，value为True/False
        torch.Size([1, 78]) torch.Size([1, 78])
        torch.Size([1, 189]) torch.Size([1, 189])
        torch.Size([1, 189]) torch.Size([1, 189])
        torch.Size([1, 21]) torch.Size([1, 21])
        torch.Size([1, 588]) torch.Size([1, 588])
        torch.Size([1, 42]) torch.Size([1, 42])
        '''

        self.data_true_available_matrix_narrow = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        self.baseline_true_available_matrix_narrow = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))

        _count, self._data_false_index_narrow = 0, []  # 存放原始的index len：750
        for key, value in self.super_index_narrowedornot_dataonly.items():
            if value==False:
                _count += 1
                self._data_false_index_narrow.append(key)
        _count, self._baseline_false_index_narrow = 0, []  # 存放原始的index len：750
        for key, value in self.super_index_narrowedornot_baselineonly.items():
            if value == False:
                _count += 1
                self._baseline_false_index_narrow.append(key)

        self._union_false_index = set(self._data_false_index_narrow).union(set(self._baseline_false_index_narrow))  # data和baseline有数据的维度的并集，narrowed index
        self._union_false_index = sorted(list(self._union_false_index))
        assert len(self._union_false_index) == len(self.narrow_index_to_name)
        # 把药物补齐（由于药物没有使用-1，_baseline_false_index_narrow和_data_false_index_narrow的drug要取并集）
        for _i, _each_normal_index in enumerate(self._union_false_index):
            _this_name = self.narrow_index_to_name[_i]
            if 'drug_day' in _this_name:
                if _each_normal_index in self._baseline_false_index_narrow and _each_normal_index not in self._data_false_index_narrow:
                    self._data_false_index_narrow.append(_each_normal_index)
                if _each_normal_index not in self._baseline_false_index_narrow and _each_normal_index in self._data_false_index_narrow:
                    self._baseline_false_index_narrow.append(_each_normal_index)
        self._data_false_index_narrow = sorted(self._data_false_index_narrow)
        self._baseline_false_index_narrow = sorted(self._baseline_false_index_narrow)

        self._intersect_false_index = set(self._data_false_index_narrow).intersection(set(self._baseline_false_index_narrow))
        self._intersect_false_index = sorted(list(self._intersect_false_index))

        # build matrix like label?
        for _i in range(self.data_true_available_matrix_narrow.shape[0]):
            for _j in range(self.data_true_available_matrix_narrow.shape[1]):
                if self.narrow_index_to_normal_index[_i] in self._data_false_index_narrow and self.narrow_index_to_normal_index[_j] in self._data_false_index_narrow:
                    self.data_true_available_matrix_narrow[_i, _j] = 1
                if self.narrow_index_to_normal_index[_i] in self._baseline_false_index_narrow and self.narrow_index_to_normal_index[_j] in self._baseline_false_index_narrow:
                    self.baseline_true_available_matrix_narrow[_i, _j] = 1

        self.intersect_true_available_matrix_narrow = self.data_true_available_matrix_narrow * self.baseline_true_available_matrix_narrow

    def build_pos_neg_basedontrueavailablematrix(self):
        '''
        self.baseline_true_available_matrix_narrow
        self.data_true_available_matrix_narrow
        self.intersect_true_available_matrix_narrow
        :return:
        '''

        self._sample_data_value_1d = np.zeros(len(self.narrow_index_to_name))
        self._sample_data_value_hasrecord = np.zeros(len(self.narrow_index_to_name))
        self._sample_data_value_slice_list = []
        self._baseline_data_value_1d = np.zeros(len(self.narrow_index_to_name))
        self._baseline_data_value_hasrecord = np.zeros(len(self.narrow_index_to_name))
        self._baseline_data_value_slice_list = []
        _name_type_helper = []
        self._intersect_false_index_narrow = np.zeros(len(self.narrow_index_to_name))
        # 利用self.narrow_index_to_name
        _index_helper = 0
        for _i in range(len(self.narrow_index_to_name)):
            _this_index_name = self.narrow_index_to_name[_i]
            if _this_index_name.startswith('temp_insulin_day'):
                _this_index_dim = 9
                _name_type_helper.append(_this_index_name)
            elif _this_index_name.startswith('insulin_day'):
                _this_index_dim = 9
                _name_type_helper.append(_this_index_name)
            elif _this_index_name.startswith('sugar_day'):
                _this_index_dim = 1
                _name_type_helper.append(_this_index_name)
            elif _this_index_name.startswith('drug_day'):
                _this_index_dim = 1
                _name_type_helper.append(_this_index_name)
            else:
                # print(_this_index_name)
                assert 'day' not in _this_index_name
                _this_index_dim = 1
                _name_type_helper.append('exam_'+_this_index_name)
            _this_index_sample_value = self.sample_insulin_data_flatten_narrow[_index_helper: _index_helper+_this_index_dim]
            _this_index_baseline_value = self.baseline_data_flatten_narrow[_index_helper: _index_helper+_this_index_dim]
            self._sample_data_value_slice_list.append(_this_index_sample_value)
            self._baseline_data_value_slice_list.append(_this_index_baseline_value)
            if _this_index_dim==9:
                self._sample_data_value_1d[_i] = _this_index_sample_value[-1]
                self._baseline_data_value_1d[_i] = _this_index_baseline_value[-1]
                self._sample_data_value_hasrecord[_i] = 1 if _this_index_sample_value[-1]!=-1 else 0
                self._baseline_data_value_hasrecord[_i] = 1 if _this_index_baseline_value[-1]!=-1 else 0
            elif _this_index_dim==1:
                self._sample_data_value_1d[_i] = _this_index_sample_value
                self._baseline_data_value_1d[_i] = _this_index_baseline_value
                self._sample_data_value_hasrecord[_i] = 1 if _this_index_sample_value != -1 else 0
                self._baseline_data_value_hasrecord[_i] = 1 if _this_index_baseline_value != -1 else 0
            _index_helper += _this_index_dim


        # 利用两个集合的差集来build self._intersect_false_index_narrow
        for _each_index in self._union_false_index:
            if _each_index not in self._intersect_false_index:
                self._intersect_false_index_narrow[self._union_false_index.index(_each_index)] = 1

        # build baseline和data交叉项的正负影响1d vector
        self.sample_data_baseline_data_common_dim_difference = (self._sample_data_value_1d - self._baseline_data_value_1d) * (1-self._intersect_false_index_narrow)  # 后面这一项去掉了两者中missing的值
        self._sample_data_baseline_data_intersect_minus_pos_neg = np.zeros(len(self.narrow_index_to_name))

        #
        for _i, (each_diff, each_name) in enumerate(
                zip(self.sample_data_baseline_data_common_dim_difference, _name_type_helper)):
            if each_name.startswith('sugar'):
                if each_diff > 0:
                    self._sample_data_baseline_data_intersect_minus_pos_neg[_i] = 1
                elif each_diff < 0:
                    self._sample_data_baseline_data_intersect_minus_pos_neg[_i] = -1


            if each_name.startswith('drug'):
                day_time_type = re.findall("\d+", each_name)  # 输出结果为列表
                type = int(day_time_type[2])
                if (each_diff>0 and type in INCREASE_SUGAR_DRUG_MASK) or (each_diff<0 and type in DECREASE_SUGAR_DRUG_MASK):
                    self._sample_data_baseline_data_intersect_minus_pos_neg[_i] = 1
                elif (each_diff<0 and type in INCREASE_SUGAR_DRUG_MASK) or (each_diff>0 and type in DECREASE_SUGAR_DRUG_MASK):
                    self._sample_data_baseline_data_intersect_minus_pos_neg[_i] = -1
                else: pass


        # # build baseline和data sugar的交叉项，存值    血檀 cosine
        self._sample_data_baseline_data_sugar_intersect_diff_value = np.zeros(len(self.narrow_index_to_name))
        for _i, (each_diff, each_name) in enumerate(zip(self.sample_data_baseline_data_common_dim_difference, _name_type_helper)):
            if each_name.startswith('sugar'):
                self._sample_data_baseline_data_sugar_intersect_diff_value[_i] = each_diff
        self._sample_data_baseline_data_sugar_intersect_diff_value_matrix = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in range(self._sample_data_baseline_data_sugar_intersect_diff_value_matrix.shape[0]):
            for _j in range(self._sample_data_baseline_data_sugar_intersect_diff_value_matrix.shape[1]):
                if self._sample_data_baseline_data_sugar_intersect_diff_value[_i] !=0 and self._sample_data_baseline_data_sugar_intersect_diff_value[_j] !=0:
                    self._sample_data_baseline_data_sugar_intersect_diff_value_matrix[_i, _j] = (self._sample_data_baseline_data_sugar_intersect_diff_value[_i] + self._sample_data_baseline_data_sugar_intersect_diff_value[_j]) / 2
        self._sample_data_baseline_data_sugar_intersect_diff_value_matrix = np.triu(self._sample_data_baseline_data_sugar_intersect_diff_value_matrix)
        self._sample_data_baseline_data_sugar_intersect_diff_value_matrix = self.strech_upper_tri_matrix_to_flatten(self._sample_data_baseline_data_sugar_intersect_diff_value_matrix)  # has intercept


        # build data和baseline 一个有，一个没有的pos neg，
        # 高血糖记录出现、低血糖记录消失、 导致 胰岛素剂量提高=>+1
        # 低血糖记录出现、高血糖记录消失 导致 胰岛素剂量降低=>-1
        # assert 药物不会出现，因为上面对药物进行了特殊处理，见“# 把药物补齐。。。”
        self._sample_data_baseline_data_nooverlap_pos_neg = np.zeros(len(self.narrow_index_to_name))
        self._sample_data_baseline_data_nooverlap_index = np.zeros(len(self.narrow_index_to_name))
        _sample_data_self_available = self._sample_data_value_1d * self._intersect_false_index_narrow
        _baseline_data_self_available = self._baseline_data_value_1d * self._intersect_false_index_narrow
        for _i, (_sampple_value, _baseline_value, _name) in enumerate(zip(_sample_data_self_available, _baseline_data_self_available, _name_type_helper)):
            if _name.startswith('drug'):
                assert _sampple_value==0 and _baseline_value==0, f"{_sampple_value} {_baseline_value}"
            if _name.startswith('sugar_day'):
                day_time = re.findall("\d+", _name)
                if _sampple_value==-1 and _baseline_value!=-1:  # 血糖记录消失
                    _status = judge_sugar_data_range(_baseline_value, int(day_time[1]))  # return low mid high
                    self._sample_data_baseline_data_nooverlap_pos_neg[_i] = {'low': 1, 'mid': 0, 'high': -1}[_status]
                    self._sample_data_baseline_data_nooverlap_index[_i] = 1
                elif _sampple_value!=-1 and _baseline_value==-1:  # 血糖记录出现
                    _status = judge_sugar_data_range(_sampple_value, int(day_time[1]))  # return low mid high
                    self._sample_data_baseline_data_nooverlap_pos_neg[_i] = {'low': -1, 'mid': 0, 'high': 1}[_status]
                    self._sample_data_baseline_data_nooverlap_index[_i] = 1
                elif _sampple_value==0 and _baseline_value==0:  # 血糖都是0
                    pass
                else:
                    print(_sampple_value, _baseline_value, _name_type_helper[_i])
                    raise ValueError


    # 全局正负限制，包括血糖、口服药，血糖一正一负的matrix
    def build_global_pos_neg_index(self):
        # build pos neg matrix
        for _i in range(len(self.narrow_index_to_name)):
            assert abs(self._sample_data_baseline_data_intersect_minus_pos_neg[_i] * self._sample_data_baseline_data_nooverlap_pos_neg[_i]) != 1
        _pos_neg_1d = self._sample_data_baseline_data_intersect_minus_pos_neg + self._sample_data_baseline_data_nooverlap_pos_neg
        self.w_pos_matrix = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        self.w_neg_matrix = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in range(len(self.narrow_index_to_name)):
            for _j in range(len(self.narrow_index_to_name)):
                if _pos_neg_1d[_i] == 1 and _pos_neg_1d[_j] == 1:
                    self.w_pos_matrix[_i, _j] = 1
                elif _pos_neg_1d[_i] == -1 and _pos_neg_1d[_j] == -1:
                    self.w_neg_matrix[_i, _j] = 1
        self.w_pos_matrix = np.triu(self.w_pos_matrix)
        self.w_pos_matrix = self.strech_upper_tri_matrix_to_flatten(self.w_pos_matrix)  # has intercept
        self.w_neg_matrix = np.triu(self.w_neg_matrix)
        self.w_neg_matrix = self.strech_upper_tri_matrix_to_flatten(self.w_neg_matrix)  # has intercept

        self.sugar_nooverlap_other_interaction_matrix = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in range(len(self.narrow_index_to_name)):
            for _j in range(len(self.narrow_index_to_name)):
                if self._sample_data_baseline_data_nooverlap_index[_i]!=0 or self._sample_data_baseline_data_nooverlap_index[_j]!=0:
                    self.sugar_nooverlap_other_interaction_matrix[_i, _j] = 1

        self.sugar_nooverlap_other_interaction_matrix = np.triu(self.sugar_nooverlap_other_interaction_matrix)
        self.sugar_nooverlap_other_interaction_matrix = self.strech_upper_tri_matrix_to_flatten(self.sugar_nooverlap_other_interaction_matrix)  # has intercept

        self.w_one_pos_one_neg_matrix = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name)))
        for _i in range(len(self.narrow_index_to_name)):
            for _j in range(len(self.narrow_index_to_name)):
                this_name_i = self.narrow_index_to_name[_i]
                this_name_j = self.narrow_index_to_name[_j]
                if 'sugar_day' in this_name_i and 'sugar_day' in this_name_j:
                    if  (_pos_neg_1d[_i] == 1 and _pos_neg_1d[_j] == -1) or (_pos_neg_1d[_i] == -1 and _pos_neg_1d[_j] == 1):
                        self.w_one_pos_one_neg_matrix[_i, _j] = 1
        self.w_one_pos_one_neg_matrix = np.triu(self.w_one_pos_one_neg_matrix)
        self.w_one_pos_one_neg_matrix = self.strech_upper_tri_matrix_to_flatten(self.w_one_pos_one_neg_matrix)

    # 与目标胰岛素注射时间的血糖
    def build_target_insulin_related_sugar_time_list(self):
        self.target_insulin_related_sugar_time = []
        # insulin time 是 0 2 4 6而sugar是 0 1 2 3 4 5 6
        for each_insulin_time, each_insulin_type in zip(self.target_insulin_time_list, self.target_insulin_type_list):
            each_insulin_type = each_insulin_type.cpu().numpy().astype(int)
            # each_insulin_name = INSULIN_DETAIL_NAME_LIST[each_insulin_type]
            each_insulin_detail = insulin_detail[each_insulin_type]
            each_insulin_class = each_insulin_detail["classes"][0]  # shot basal premix
            if each_insulin_class == 'shot':  # 第一次完成
                if each_insulin_time in [0, 1]:
                    self.target_insulin_related_sugar_time.append([1])
                elif each_insulin_time in [2, 3]:
                    self.target_insulin_related_sugar_time.append([3])
                elif each_insulin_time in [4, 5]:
                    self.target_insulin_related_sugar_time.append([5])
                elif each_insulin_time in [6]:
                    self.target_insulin_related_sugar_time.append([6])
                else:
                    raise ValueError
            elif each_insulin_class == 'basal':  # 第一次完成
                if each_insulin_time in [0, 1]:
                    self.target_insulin_related_sugar_time.append([1, 2, 3, 4, 5, 6])
                elif each_insulin_time in [2, 3]:
                    self.target_insulin_related_sugar_time.append([3, 4, 5, 6])
                elif each_insulin_time in [4, 5]:
                    self.target_insulin_related_sugar_time.append([5, 6])
                elif each_insulin_time in [6]:
                    self.target_insulin_related_sugar_time.append([0])
                else:
                    raise ValueError
            elif each_insulin_class == 'premix':  # 第一次完成
                if each_insulin_time in [0, 1]:
                    self.target_insulin_related_sugar_time.append([1, 3])
                elif each_insulin_time in [2, 3]:
                    self.target_insulin_related_sugar_time.append([3, 5])
                elif each_insulin_time in [4, 5]:
                    self.target_insulin_related_sugar_time.append([5, 0, 2])
                elif each_insulin_time in [6]:
                    self.target_insulin_related_sugar_time.append([6, 0, 1])
                else:
                    raise ValueError


    def build_time_related_index(self):
        _time_related_index_list = [[] for _ in range(len(self.target_insulin_time_list))]
        _time_related_sugar_index_list = [[] for _ in range(len(self.target_insulin_time_list))]
        print("Warning, need update 睡前血糖关系")

        for _i in range(len(self.narrow_index_to_name)):
            this_name = self.narrow_index_to_name[_i]
            # 处理与target时间相关的内容
            if 'temp_insulin' in this_name or 'insulin_' in this_name or 'sugar' in this_name or 'drug' in this_name:
                day_time = re.findall("\d+", this_name)  # 输出结果为列表
                day = int(day_time[0])
                time = int(day_time[1])
                if int(time/2)*2 in self.target_insulin_time_list:
                    index_of_time = self.target_insulin_time_list.index(int(time/2)*2)
                    _time_related_index_list[index_of_time].append(_i)
            if 'sugar' in this_name:
                day_time = re.findall("\d+", this_name)  # 输出结果为列表
                day = int(day_time[0])
                time = int(day_time[1])  # 血糖检测时间
                for inject_index, each_related_sugar_list in enumerate(self.target_insulin_related_sugar_time):
                    if time in each_related_sugar_list:
                        _time_related_sugar_index_list[inject_index].append(_i)

        # build 目标时间各项因素的interaction
        self.matrix_target_time_interaction_index_list = [
            np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name))) for _ in
            range(len(self.target_insulin_time_list))]
        for _ii, each_time_related_index in enumerate(_time_related_index_list):
            for _i in each_time_related_index:
                for _j in each_time_related_index:
                    self.matrix_target_time_interaction_index_list[_ii][_i, _j] = 1
        for _i, each in enumerate(self.matrix_target_time_interaction_index_list):
            np.save(f'tmp_matrix_target_time_interaction_index_list{_i}.npy', each)
        self.matrix_target_time_interaction_index_list = [np.triu(each) for each in
                                                          self.matrix_target_time_interaction_index_list]
        self.matrix_target_time_interaction_index_list = [self.strech_upper_tri_matrix_to_flatten(each) for each
                                                          in self.matrix_target_time_interaction_index_list]
        # build 目标时间血糖的interaction
        self.matrix_target_time_sugar_interaction_index_list = [
            np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name))) for _ in
            range(len(self.target_insulin_time_list))]
        for _ii, each_time_related_sugar_index in enumerate(_time_related_sugar_index_list):
            for _i in each_time_related_sugar_index:
                for _j in each_time_related_sugar_index:
                    self.matrix_target_time_sugar_interaction_index_list[_ii][_i, _j] = 1
        for _i, each in enumerate(self.matrix_target_time_sugar_interaction_index_list):
            np.save(f'tmp_matrix_target_time_sugar_interaction_index_list{_i}.npy', each)
        self.matrix_target_time_sugar_interaction_index_list = [np.triu(each) for each in
                                                                self.matrix_target_time_sugar_interaction_index_list]
        self.matrix_target_time_sugar_interaction_index_list = [self.strech_upper_tri_matrix_to_flatten(each)
                                                                for each in
                                                                self.matrix_target_time_sugar_interaction_index_list]

    def fit(self, X: torch.tensor, y: torch.tensor, sample_weight=None) -> None:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(torch.float32)
        if isinstance(sample_weight, np.ndarray):
            sample_weight = torch.from_numpy(sample_weight).to(torch.float32)

        X = X.cuda()
        y = y.cuda()
        sample_weight = sample_weight.cuda()

        X = X.rename(None)
        y = y.rename(None).squeeze().unsqueeze(1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if len(y.shape)==3 and y.shape[1]==1:
            y = y.squeeze(dim=1)
        assert (len(y.shape) == 2)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1).to(X.device), X], dim=1)

        if sample_weight is not None:
            # print(sample_weight.shape)  # torch.Size([600])
            # print(torch.sqrt(sample_weight).shape)  # torch.Size([600])
            W2 = torch.sqrt(sample_weight).squeeze().unsqueeze(1)
            # print(W2.shape)  # torch.Size([600, 1])
            y = y * W2
            X = X * W2

        lhs = X
        rhs = y
        # del X, y, sample_weight

        if self.init_method == 'random':
            self.w = torch.rand((lhs.shape[1], rhs.shape[1]))
            self.w = Variable(self.w.data).to(X.device, X.dtype)
            self.w.requires_grad = True
        elif self.init_method == 'analytic':
            self.w = torch.linalg.lstsq(X.T @ X, X.T @ y).solution + 0 * torch.rand((lhs.shape[1], rhs.shape[1])).to(X.device, X.dtype)
            self.w = Variable(self.w.data).to(X.device, X.dtype)
            self.w.requires_grad = True
        else:
            raise NotImplementedError

        optimize_epoch = 500

        ep_loss_list = []
        start_time = time.time()
        for epoch in range(optimize_epoch):
            epoch_output_info = ''
            loss = self.mse_loss(torch.matmul(lhs, self.w), rhs)
            epoch_output_info += f'mse error {loss.data.item()}, '
            if self.w_l1 != 0:
                loss_l1_regularization = self.w_l1 * torch.norm(self.w[1:], p=1)
                loss += loss_l1_regularization
                epoch_output_info += f'w l1 {loss_l1_regularization.data.item()}, '
            if self.w_l2 != 0:
                loss_l2_regularization = self.w_l2 * torch.norm(self.w[1:], p=2)
                loss += loss_l2_regularization
                epoch_output_info += f'w l2 {loss_l2_regularization.data.item()}, '
            if self.insulin_l1 != 0:
                loss_insulin_l1 = self.insulin_l1 * torch.norm(
                    self.w[1:] * self.matrix_insulin_interaction_index.to(self.w.device, self.w.dtype)[1:], p=1)
                loss += loss_insulin_l1
                epoch_output_info += f'insulin l1 {loss_insulin_l1.data.item()}, '
            if self.insulin_l2 != 0:
                loss_insulin_l2 = self.insulin_l2 * torch.norm(
                    self.w[1:] * self.matrix_insulin_interaction_index.to(self.w.device, self.w.dtype)[1:], p=2)
                loss += loss_insulin_l2
                epoch_output_info += f'insulin l2 {loss_insulin_l2.data.item()}, '
            if self.exam_l1 != 0:
                loss_exam_l1 = self.exam_l1 * torch.norm(
                    self.w[1:] * self.matrix_exam_interaction_index.to(self.w.device, self.w.dtype)[1:], p=1)
                loss += loss_exam_l1
                epoch_output_info += f'exam l1 {loss_exam_l1.data.item()}, '
            if self.exam_l2 != 0:
                loss_exam_l2 = self.exam_l2 * torch.norm(
                    self.w[1:] * self.matrix_exam_interaction_index.to(self.w.device, self.w.dtype)[1:], p=2)
                loss += loss_exam_l2
                epoch_output_info += f'exam l2 {loss_exam_l2.data.item()}, '


            # 全局正负
            loss_global_pos = torch.norm(torch.relu(-(self.w[1:] * self.w_pos_matrix.to(self.w.device, self.w.dtype)[1:])))
            loss_global_neg = torch.norm(torch.relu( (self.w[1:] * self.w_neg_matrix.to(self.w.device, self.w.dtype)[1:])))
            loss += loss_global_pos
            loss += loss_global_neg
            epoch_output_info += f'sugar pos {loss_global_pos.data.item()} neg {loss_global_neg.data.item()}, '

            # 胰岛素 血糖 interaction 正负
            loss_insulin_sugar_pos_neg = torch.norm(torch.relu(-(self.w[1:] * self.insulin_sugar_interaction_time_related_pos_neg_matrix.to(self.w.device, self.w.dtype)[1:])))
            loss += loss_insulin_sugar_pos_neg
            epoch_output_info += f'inssug posneg {loss_insulin_sugar_pos_neg.data.item()}, '

            w_square = torch.norm(self.w[1:], p=2, dim=0) ** 2  # torch.Size([4])


            first_order_square = torch.norm(self.w[1:] * self.matrix_first_order_index[1:].to(self.w.device, self.w.dtype), p=2, dim=0) ** 2
            second_order_square = torch.norm(self.w[1:] * self.matrix_second_order_index[1:].to(self.w.device, self.w.dtype), p=2, dim=0) ** 2


            # 时间相关的血糖
            epoch_output_info += 'timerelatedsugar / '
            for _ii, each_time_sugar_interaction_matrix in enumerate(self.matrix_target_time_sugar_interaction_index_list):
                each_time_sugar_interaction_square = torch.norm(self.w[1:][:, _ii].reshape(-1, 1) * each_time_sugar_interaction_matrix[1:].to(self.w.device, self.w.dtype), p=2, dim=0) ** 2
                loss_each_time_sugar_interaction_matrix = torch.norm(torch.relu(-(each_time_sugar_interaction_square / w_square[_ii] - 0.25)))
                loss += loss_each_time_sugar_interaction_matrix
                epoch_output_info += '%.3f ' % loss_each_time_sugar_interaction_matrix.data.item()



            # 二阶的pos neg要与一阶相同，要先build matrix
            self.first_order_w = self.w[1:] * self.matrix_first_order_index[1:].to(self.w.device, self.w.dtype)  # flattened, torch.Size([1225, 4])
            this_time_first_order_matrix = np.zeros((len(self.narrow_index_to_name), len(self.narrow_index_to_name), self.first_order_w.shape[1]))
            for _ind in range(len(self.narrow_index_to_name)):
                this_time_first_order_matrix[_ind, _ind, :] = self.first_order_w[_ind * len(self.narrow_index_to_name) - int((0+_ind) * (_ind+1) / 2) + _ind, :].detach().cpu().numpy()
            this_time_first_order_matrix = np.sign(this_time_first_order_matrix)
            this_time_first_order_matrix = np.sign(this_time_first_order_matrix)
            for _k in range(self.first_order_w.shape[1]):
                for _i in range(this_time_first_order_matrix.shape[0]):
                    for _j in range(this_time_first_order_matrix.shape[1]):
                        if this_time_first_order_matrix[_i, _i, _k]==this_time_first_order_matrix[_j, _j, _k]:
                            this_time_first_order_matrix[_i, _j, _k] = this_time_first_order_matrix[_i, _i, _k]
            _tmp = []
            for _i in range(self.first_order_w.shape[1]):
                _tmp.append(self.strech_upper_tri_matrix_to_flatten(np.triu(this_time_first_order_matrix[:, :, _i])))
            self.matrix_second_order_follow_first_order_pos_neg = torch.hstack(_tmp)  # (1226, 4)
            loss_second_follow_first = torch.norm(torch.relu(-self.w[1:] * self.matrix_second_order_follow_first_order_pos_neg[1:].to(self.w.device, self.w.dtype)))
            loss += loss_second_follow_first
            epoch_output_info += f'scd_folw_fst posneg {loss_second_follow_first.data.item()}, '


            grad = torch.autograd.grad(loss, self.w, retain_graph=False)[0]
            grad = grad / torch.norm(grad)
            self.w = self.w - self.lr * grad
            if epoch % 50 == 0:
                print(epoch_output_info)
            ep_loss_list.append(loss.data.item())
        print('time', time.time() - start_time)
        print(self.mse_loss(torch.matmul(lhs, self.w), rhs))


        print('Warning alpha == 0 mse_f of residual', self.mse_loss(torch.matmul(lhs, self.w), rhs))


        if self.fit_intercept:
            self.intercept_ = self.w[0].detach().cpu()
            self.coef_ = self.w[1:]
        else:
            self.coef_ = self.w
        self.coef_ = self.coef_.t().detach().cpu()
        print('self.coef_.shape', self.coef_.shape)  # torch.Size([7, 16836])

        return epoch_output_info

    def predict(self, X: torch.tensor) -> None:
        # X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1).to(X.device), X], dim=1)
        # print(X, self.w)
        return (X @ self.w.to(torch.float)).detach()


    def translate_judge_type(self, _narrow_row_num, _narrow_col_num=None,
                             f_out=None, drug_detail_name_list=None):
        _name_row = self.narrow_index_to_name[_narrow_row_num]
        _name_col = None
        if _narrow_col_num is not None:
            _name_col = self.narrow_index_to_name[_narrow_col_num]
        if _name_row.startswith('insulin_day'):
            print('【胰岛素', file=f_out, end='')
        elif _name_row.startswith('temp_insulin_day'):
            print('【胰岛素临时注射', file=f_out, end='')
        elif _name_row.startswith('sugar_day'):
            print('【血糖', file=f_out, end='')
        elif _name_row.startswith('drug_day'):
            drug_index = int(re.findall('\d+', _name_row)[2])
            if drug_index in INCREASE_SUGAR_DRUG_MASK:
                print('【口服升糖药', file=f_out, end='')
            elif drug_index in DECREASE_SUGAR_DRUG_MASK:
                print('【口服降糖药', file=f_out, end='')
            else:
                raise NotImplementedError
        elif _name_row.startswith('days'):
            print('【时间因素', file=f_out, end='')
        else:  # 检查
            if _name_row in JIBENXINXI:
                print('【基本信息', file=f_out, end='')
            elif _name_row in HUAYAN_YIDAOGONGNENG:
                print('【化验-胰岛功能', file=f_out, end='')
            elif _name_row in HUAYAN_XUESHENGHUA:
                print('【化验-血生化', file=f_out, end='')
            elif _name_row in HUAYAN_YIDAOSUZISHENKANGTI:
                print('【化验-胰岛素自身抗体', file=f_out, end='')
            elif _name_row in HUAYAN_JISU:
                print('【化验-激素', file=f_out, end='')
            elif _name_row in HUAYAN_NIAOSHENGHUA:
                print('【化验-尿生化', file=f_out, end='')
            else:
                print(_name_row)
                raise ValueError
        if _name_col is not None:
            if _name_col.startswith('insulin_day'):
                print('，胰岛素', file=f_out, end='')
            elif _name_col.startswith('temp_insulin_day'):
                print('，胰岛素临时注射', file=f_out, end='')
            elif _name_col.startswith('sugar_day'):
                print('，血糖', file=f_out, end='')
            elif _name_col.startswith('drug_day'):
                if _narrow_col_num in self._drug_increase_sugar_index:
                    print('，口服升糖药', file=f_out, end='')
                elif _narrow_col_num in self._drug_decrease_sugar_index:
                    print('，口服降糖药', file=f_out, end='')
            elif _name_col.startswith('days'):
                print('，时间因素', file=f_out, end='')
            else:  # 检查
                if _name_col in JIBENXINXI:
                    print('，基本信息', file=f_out, end='')
                elif _name_col in HUAYAN_YIDAOGONGNENG:
                    print('，化验-胰岛功能', file=f_out, end='')
                elif _name_col in HUAYAN_XUESHENGHUA:
                    print('，化验-血生化', file=f_out, end='')
                elif _name_col in HUAYAN_YIDAOSUZISHENKANGTI:
                    print('，化验-胰岛素自身抗体', file=f_out, end='')
                elif _name_col in HUAYAN_JISU:
                    print('，化验-激素', file=f_out, end='')
                elif _name_col in HUAYAN_NIAOSHENGHUA:
                    print('，化验-尿生化', file=f_out, end='')
                else:
                    print(_name_col)
                    raise ValueError
        print('】', file=f_out, end=' ')

    def translate(self, _narrow_row_num, _narrow_col_num, each, _min_index, _predict_baseline_diff, f_out=None):
        if _narrow_row_num==_narrow_col_num:
            print('单一原因', end=' ', file=f_out)
            self.translate_judge_type(_narrow_row_num, f_out=f_out)
            print(each.reshape(-1)[_min_index])
            print('<影响%.2f %2d' % (each.reshape(-1)[_min_index], each.reshape(-1)[_min_index]/abs(_predict_baseline_diff)*100), end='%> {', file=f_out)
            print(self.translation_sentence_list[_narrow_col_num], end=' ', file=f_out)
            print('}', file=f_out)
        else:
            print('复合原因', end=' ', file=f_out)
            self.translate_judge_type(_narrow_row_num, _narrow_col_num=_narrow_col_num, f_out=f_out)
            print('<影响%.2f %2d' % (each.reshape(-1)[_min_index], each.reshape(-1)[_min_index]/abs(_predict_baseline_diff)*100), end='%> {', file=f_out)
            print(self.translation_sentence_list[_narrow_row_num], '}{', self.translation_sentence_list[_narrow_col_num], end=' ', file=f_out)
            print('}', file=f_out)

    def build_translation_sentence_renhua(self, index_name, index_num, return_type=True, disappear=False):  # return 人话
        if '_day' in index_name:
            day_time_x = re.findall("\d+", index_name)
            day = int(day_time_x[0])
            time = int(day_time_x[1])
            if 'insulin' in index_name:
                # insulin_type = int(day_time_x[2])
                if disappear:
                    insulin_data = self._baseline_data_value_slice_list[index_num]
                else:
                    insulin_data = self._sample_data_value_slice_list[index_num]
                insulin_type = insulin_data[0].cpu().numpy().astype(int)
                insulin_type = INSULIN_DETAIL_NAME_LIST[insulin_type]
                if return_type:
                    return_info = f'前{2 - day}天{TIME_NAME[time]}{insulin_type}'
                else:
                    return_info = f'前{2 - day}天{TIME_NAME[time]}'
                return return_info
            elif 'sugar' in index_name:
                return_info = f'前{2 - day}天{TIME_NAME[time]}血糖'
                return return_info
            elif 'drug' in index_name:
                drug_type = int(day_time_x[2])
                drug_type = DRUG_DETAIL_NAME_LIST[drug_type]
                return_info = f'前{2 - day}天{TIME_NAME[time]}药物{drug_type}'
                return return_info
            else:
                raise ValueError
        else:  # examination
            return index_name

    def value_equal(self, data1, data2):
        return data1.equal(data2)

    def build_translation_sentence(self):
        # self._sample_data_value_1d
        # self._baseline_data_value_1d
        self.translation_sentence_list = []
        for _i in range(len(self.narrow_index_to_name)):
            _this_index_name = self.narrow_index_to_name[_i]
            _baseline_hasrecord = self._baseline_data_value_hasrecord[_i]
            _sample_hasrecord = self._sample_data_value_hasrecord[_i]
            _baseline_value = self._baseline_data_value_slice_list[_i]
            _sample_value = self._sample_data_value_slice_list[_i]
            if _baseline_hasrecord==1 and _sample_hasrecord==0: # 消失
                self.translation_sentence_list.append(self.build_translation_sentence_renhua(_this_index_name, _i, disappear=True)+'记录缺失')
            elif _baseline_hasrecord==0 and _sample_hasrecord==1: # 出现
                self.translation_sentence_list.append(self.build_translation_sentence_renhua(_this_index_name, _i)+'记录（从无到有）')
            elif _baseline_hasrecord == 1 and _sample_hasrecord == 1:  # 是重叠维度
                if self.value_equal(_sample_value, _baseline_value):
                    self.translation_sentence_list.append(self.build_translation_sentence_renhua(_this_index_name, _i)+'值(未变化)')
                else:
                    if 'sugar' in _this_index_name:
                        _midname = '值'
                        self.translation_sentence_list.append(
                            self.build_translation_sentence_renhua(_this_index_name, _i)
                            + _midname
                            + '由%.2f变为%.2f' % (_baseline_value.cpu().numpy()[-1],
                                               _sample_value.cpu().numpy()[-1]
                                               ))
                    elif 'insulin' in _this_index_name:
                        _midname = '剂量'
                        if INSULIN_DETAIL_NAME_LIST[int(_baseline_value.cpu().numpy()[0])] == INSULIN_DETAIL_NAME_LIST[int(_sample_value.cpu().numpy()[0])]:
                            self.translation_sentence_list.append(
                                self.build_translation_sentence_renhua(_this_index_name, _i, return_type=False)
                                + '由%s%s%.2f变为%s%.2f' % (
                                INSULIN_DETAIL_NAME_LIST[int(_baseline_value.cpu().numpy()[0])].replace(" ", ""),
                                _midname,
                                _baseline_value.cpu().numpy()[-1],
                                _midname,
                                _sample_value.cpu().numpy()[-1]
                                ))
                        else:
                            self.translation_sentence_list.append(self.build_translation_sentence_renhua(_this_index_name, _i, return_type=False)
                                                                  + '由%s%s%.2f变为%s%s%.2f' % (INSULIN_DETAIL_NAME_LIST[int(_baseline_value.cpu().numpy()[0])].replace(" ", ""),
                                                                                            _midname,
                                                                                        _baseline_value.cpu().numpy()[-1],
                                                                                        INSULIN_DETAIL_NAME_LIST[int(_sample_value.cpu().numpy()[0])].replace(" ", ""),
                                                                                             _midname,
                                                                                        _sample_value.cpu().numpy()[-1]
                                                                                           ))
                    elif 'drug' in _this_index_name:
                        _midname = '剂量'
                        drug_type = int(re.findall("\d+", _this_index_name)[2])
                        self.translation_sentence_list.append(
                            self.build_translation_sentence_renhua(_this_index_name, _i)
                            + _midname
                            + '由%.2f变为%.2f' % (_baseline_value.cpu().numpy(),
                                                     _sample_value.cpu().numpy()
                                                     ))
                    else:
                        print(_this_index_name)
                        assert 1==0
            else:
                raise ValueError
        assert len(self.translation_sentence_list)==len(self.narrow_index_to_name)

        # extend translation_sentence_list to translation_sentence_list_full
        self.translation_sentence_list_full = []
        _narrow_index_pointer = 0
        _super_index_pointer = 0
        while _super_index_pointer < len(self.super_index_to_name) and _narrow_index_pointer < len(self.narrow_index_to_name):
            # print(_narrow_index_pointer, _super_index_pointer, self.narrow_index_to_name[_narrow_index_pointer], self.super_index_to_name[_super_index_pointer])
            if self.narrow_index_to_name[_narrow_index_pointer] == self.super_index_to_name[_super_index_pointer]:
                self.translation_sentence_list_full.append(self.translation_sentence_list[_narrow_index_pointer])
                _narrow_index_pointer += 1
                _super_index_pointer += 1
            else:
                self.translation_sentence_list_full.append("")
                _super_index_pointer += 1
        while len(self.translation_sentence_list_full) < len(self.super_index_to_name):
            self.translation_sentence_list_full.append("")


    def strech_upper_tri_matrix_to_flatten(self, matrix):
        flatten = []
        for _row_index in range(matrix.shape[0]):
            flatten.append(matrix[_row_index, _row_index:])
        flatten = torch.from_numpy(np.hstack(flatten))  # torch.Size([1653])
        if self.fit_intercept:
            flatten = torch.hstack([torch.zeros(1), flatten]).reshape(-1, 1)
        return flatten


def build_all_missing_baseline(example_batch_data):
    example_sugar_data, example_insulin_data, y_sugar, y_insulin = example_batch_data
    example_sugar_data = list(example_sugar_data)
    example_insulin_data = list(example_insulin_data)
    batch_size = example_sugar_data[0].shape[0]
    print('build_all_missing_baseline batch size', batch_size)
    # format batch size = 1
    for i in range(len(example_sugar_data)):
        if torch.is_tensor(example_sugar_data[i]):
            example_sugar_data[i] = example_sugar_data[i][0].unsqueeze(0)
        elif type(example_sugar_data[i])==list and len(example_sugar_data[i])==0:
            pass
        else:
            raise ValueError
    for i in range(len(example_insulin_data)):
        if torch.is_tensor(example_insulin_data[i]):
            example_insulin_data[i] = example_insulin_data[i][0].unsqueeze(0)
        elif type(example_insulin_data[i])==list and len(example_insulin_data[i])==0:
            pass
        else:
            raise ValueError
    # format done
    # change value to default missing value according to decided rule
    example_sugar_data[0] = torch.ones_like(example_sugar_data[0]) * -1  # examination,
    example_sugar_data[1] = torch.ones_like(example_sugar_data[1]) * -1  # insulin,
    example_sugar_data[2] = torch.ones_like(example_sugar_data[2]) * -1  # temp_insulin,
    example_sugar_data[3] = torch.ones_like(example_sugar_data[3]) * -1  # sugar,
    example_sugar_data[4] = torch.ones_like(example_sugar_data[4]) * 0  # drug,
    example_sugar_data[5] = torch.ones_like(example_sugar_data[5]) * -1  # days,
    example_sugar_data[6] = example_sugar_data[6]  # empty mask
    example_insulin_data[0] = torch.ones_like(example_insulin_data[0]) * -1  # examination,
    example_insulin_data[1] = torch.ones_like(example_insulin_data[1]) * -1  # insulin,
    example_insulin_data[2] = torch.ones_like(example_insulin_data[2]) * -1  # temp_insulin,
    example_insulin_data[3] = torch.ones_like(example_insulin_data[3]) * -1  # sugar,
    example_insulin_data[4] = torch.ones_like(example_insulin_data[4]) * 0  # drug,
    example_insulin_data[5] = torch.ones_like(example_insulin_data[5]) * -1  # days,
    example_insulin_data[6] = example_insulin_data[6]  # empty mask

    return [example_sugar_data, example_insulin_data]

def build_average_baseline():
    pass



# Driver code
if __name__ == "__main__":
    n = 392
    r = 10

    printNcR(n, r)
