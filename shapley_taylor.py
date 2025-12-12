import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import copy

from torch.utils.data import TensorDataset, DataLoader
from itertools import combinations
import shapley_utils
from shapley_utils import printNcR, C, TorchRidge, Constrained_least_square, GradientDescendRidge, GradientDescendRidge_onlysugarposneg
from tqdm import tqdm
from scipy.special import comb
# import shap
import matplotlib.pyplot as plt
from datasets.utils import ExaminationEncoder
import time

class ShapleyTaylor:
    def __init__(self, F, N):
        self.F = F
        self.N = N
        self.N_flatten = self.convert_data_to_flatten(N)
        self.n = len(self.N_flatten)

    def assign_baseline_data(self, data):
        self.template_data = data
        self.template_data_flattern = self.convert_data_to_flatten(data)

    def convert_data_to_flatten(self, data):
        if type(data) == tuple or type(data) == list:
            raise NotImplementedError
        elif type(data) == int:
            print(data)
            assert 1 == 0
        elif type(data) == torch.Tensor:
            return torch.flatten(data)
        else:
            print(data)
            assert 1 == 0

    def convert_flatten_to_data(self, flatten):
        return torch.reshape(flatten, self.template_data.shape)

    def return_all_permutation(self, father_mask):
        ans = []
        non_zero_index = np.where(father_mask == 1)[0]
        subset_index = sum([list(map(list, combinations(non_zero_index, i))) for i in range(len(non_zero_index) + 1)],
                           [])
        for each_index in subset_index:
            ans.append(self.mask_by_index_1to0(each_index, father_mask=father_mask))
        assert len(ans) == 2 ** torch.sum(father_mask)
        return ans

    def mask_by_complementary_index_1to0(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.ones_like(self.template_data_flattern)
        else:
            mask = copy.deepcopy(father_mask)
        # print('index', index)
        for i in range(len(mask)):
            if i not in index:
                mask[i] = 0
        return mask

    def mask_by_index_1to0(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.ones_like(self.template_data_flattern)
        else:
            mask = copy.deepcopy(father_mask)
        # print('mask_by_index_1to0 before mask', mask, father_mask)
        for i in index:
            # print('mask_by_index_1to0 i', i)
            mask[i] = 0
        # print('mask_by_index_1to0 after mask', mask, father_mask)
        return mask

    def mask_to_data(self, mask_flatten, data__flatten):
        return mask_flatten * data__flatten + (1 - mask_flatten) * self.template_data_flattern

    def return_T_S_pair(self, k):
        N_index = range(self.n)
        S_index_k_order_list = list(map(list, combinations(N_index, k)))
        # print('k, S_index_k_order_list', k, S_index_k_order_list)
        S_mask_flatten_list = []
        for each in S_index_k_order_list:
            temp_flatten_S = self.mask_by_complementary_index_1to0(each)
            # print('temp_flatten_S', temp_flatten_S)
            assert torch.sum(temp_flatten_S) == k  # to be del
            S_mask_flatten_list.append(temp_flatten_S)
        # print('S_mask_flatten_list', S_mask_flatten_list)

        T_mask_flatten_list = []
        for each_S_mask in S_mask_flatten_list:
            temp_T_mask = self.return_all_permutation(torch.ones_like(each_S_mask) - each_S_mask)
            T_mask_flatten_list.append(temp_T_mask)
        # print('T_mask_flatten_list', T_mask_flatten_list)

        return T_mask_flatten_list, S_mask_flatten_list

    def delta_S_F_T(self, S_mask, T_mask):
        sum = 0
        W_mask_list = self.return_all_permutation(S_mask)
        for W_mask in W_mask_list:
            mask = W_mask + T_mask
            data = self.N_flatten * mask + self.template_data_flattern * (1 - mask)
            data = data.reshape(1, -1)
            sum += (-1) ** (torch.sum(W_mask) - torch.sum(S_mask)) * self.F(data)
        return sum

    def calculate_shapley_taylor(self, max_k):
        ans = []
        for k in range(1, max_k):
            T_mask_list, S_mask_list = self.return_T_S_pair(k)
            for each_T, each_S in zip(T_mask_list, S_mask_list):
                summation = 0
                for each_each_T in each_T:
                    summation += self.delta_S_F_T(each_S, each_each_T) / C(self.n - 1,
                                                                           int(torch.sum(each_each_T).numpy()),
                                                                           method='dc')

                I_S_k_F = k / self.n * summation
                ans.append(I_S_k_F.data)
                # print('k', k, each_S, I_S_k_F.data)
        return torch.Tensor(ans)


class GroupShapleyTaylor:
    def __init__(self, F, sample_input_tuple):
        sample_sugar_data, sample_insulin_data = sample_input_tuple
        sample_sugar_data, sample_insulin_data = list(sample_sugar_data), list(sample_insulin_data)
        self.device = sample_sugar_data[0].device
        self.F = F.to(self.device)

        self._zero_tensor_mask = None
        self._one_tensor_mask = None
        self.N = 0
        self.size_list = None
        self.super_index_to_index = {}
        # init self._zero_tensor_mask, self.N self.size_list do not modify
        self.init_group_mask(sample_sugar_data)
        self._zero_tensor_mask_flatten = self.convert_data_to_flatten(self._zero_tensor_mask)
        self._one_tensor_mask_flatten = self.convert_data_to_flatten(self._one_tensor_mask)

        self.sample_insulin_data_flatten = self.convert_data_to_flatten(sample_insulin_data)
        # sample_insulin_data = self.convert_flatten_to_data(sample_insulin_data_flatten)

        # self.n = len(self.N_flatten)
        # self.group_mask = None

    def create_list_of_zero_tensor(self, list_of_tensor_size, batch=0):
        res = []
        if batch == 0:
            for each in list_of_tensor_size:
                res.append(torch.zeros(each))
            raise NotImplementedError
        else:
            for each in list_of_tensor_size:
                if type(each) == list and len(each) > 0:
                    res.append(torch.zeros(each[1:]).unsqueeze(dim=0).repeat(batch, 1, 1))
                elif type(each) == list and len(each) == 0:
                    pass
                else:
                    raise NotImplementedError
        # for each in res:
        #     print('create_list_of_zero_tensor', each.shape)
        return res

    def create_list_of_one_tensor(self, list_of_tensor_size, batch=0):
        res = []
        if batch == 0:
            for each in list_of_tensor_size:
                res.append(torch.ones(each))
            raise NotImplementedError
        else:
            for each in list_of_tensor_size:
                if type(each) == list and len(each) > 0:
                    res.append(torch.ones(each[1:]).unsqueeze(dim=0).repeat(batch, 1, 1))
                elif type(each) == list and len(each) == 0:
                    pass
                else:
                    raise NotImplementedError
        # for each in res:
        #     print('create_list_of_one_tensor', each.shape)
        return res

    def multiplyList(self, myList):

        result = 1
        for x in myList:
            result = result * x
        return result

    def init_group_mask(self, data):
        if type(data) == tuple or type(data) == list:
            self.size_list = []
            self.size_list_product = []
            # self.group_mask = []
            # init self._zero_tensor_mask
            for each in data:
                if torch.is_tensor(each):
                    _shape = list(each.shape)
                    _shape[0] = 1
                    self.size_list.append(_shape)
                    self.size_list_product.append(self.multiplyList(_shape))
                elif type(each) == list and len(each) == 0:
                    self.size_list.append([])
                    self.size_list_product.append(0)
                else:
                    print(each)
                    raise NotImplementedError
            self._zero_tensor_mask = self.create_list_of_zero_tensor(self.size_list, batch=1)
            self._one_tensor_mask = self.create_list_of_one_tensor(self.size_list, batch=1)
            # init N & super_index_to_index
            _super_index_counter = 0
            _index_counter = 0
            for each in self._zero_tensor_mask:
                if torch.is_tensor(each):
                    if each.shape[1] == 1:
                        print(each.shape[2])
                        self.N += each.shape[2]
                        for _i in range(each.shape[2]):
                            self.super_index_to_index[_super_index_counter] = (_index_counter, _index_counter + 1)
                            _super_index_counter += 1
                            _index_counter += 1
                    else:
                        # print(each.shape[1])
                        self.N += each.shape[1]
                        for _i in range(each.shape[1]):
                            self.super_index_to_index[_super_index_counter] = (
                            _index_counter, _index_counter + each.shape[2])
                            _super_index_counter += 1
                            _index_counter += +each.shape[2]
                elif type(each) == list and len(each) == 0:
                    pass
                else:
                    raise ValueError
            # print(self.super_index_to_index)
            # print(self.N)
        else:
            print(data)
            raise NotImplementedError

    def convert_data_to_flatten(self, data):
        if type(data) == tuple or type(data) == list:
            if type(data) == tuple:
                data = list(data)
            for _i in range(len(data)):
                if torch.is_tensor(data[_i]):
                    # print(data[_i].shape[0])
                    data[_i] = data[_i].reshape(data[_i].shape[0], -1)
                elif type(data[_i]) == list and len(data[_i]) == 0:
                    data.remove([])  # 如果不是最后一个是list可能会有问题
                else:
                    print(data[_i])
                    print(type(data[_i]))
                    raise NotImplementedError
            # assert 1==0
            data = torch.hstack(data)
        elif type(data) == int:
            print(data)
            raise NotImplementedError
        elif type(data) == torch.Tensor:
            raise NotImplementedError
        else:
            print(data)
            raise NotImplementedError
        # for each in data:
        #     if torch.is_tensor(each):
        #         print('to flatten', each.shape)
        #     else:
        #         print('to flatten', each)
        return data

    def size_splits(self, tensor, split_size_product, split_sizes, dim=0):
        """Splits the tensor according to chunks of split_sizes.

        Arguments:
            tensor (Tensor): tensor to split.
            split_size_product: product of size
            split_sizes (list(int)): sizes of chunks
            dim (int): dimension along which to split the tensor.
        """
        _batch = tensor.shape[0]
        if dim < 0:
            dim += tensor.dim()

        dim_size = tensor.size(dim)
        if dim_size != torch.sum(torch.Tensor(split_size_product)):
            raise KeyError("Sum of split sizes exceeds tensor dim")

        splits = torch.cumsum(torch.Tensor([0] + split_size_product), dim=0)[:-1]
        ans = []
        for start, length, shape in zip(splits, split_size_product, split_sizes):
            if len(shape) == 0:
                ans.append([])
            else:
                shape[0] = _batch
                ans.append(tensor.narrow(int(dim), int(start), int(length)).reshape(shape))

        return ans

    def convert_flatten_to_data(self, flatten_tensor):

        ans = self.size_splits(flatten_tensor, self.size_list_product, self.size_list, dim=1)

        return ans

    def assign_baseline_data(self, data):
        self.baseline_data = data
        self.baseline_data_flatten = self.convert_data_to_flatten(data)

    # def random_combination(self, iterable, r):
    #     "Random selection from itertools.combinations(iterable, r)"
    #     pool = tuple(iterable)
    #     n = len(pool)
    #     indices = sorted(random.sample(xrange(n), r))
    #     return tuple(pool[i] for i in indices)

    def return_all_permutation(self, father_mask, T_sample_limit=None):
        if T_sample_limit is None:  # slow algorithm, return all permutation
            ans = []
            non_zero_index = np.where(father_mask == 1)[0]
            subset_index = sum(
                [list(map(list, combinations(non_zero_index, i))) for i in range(len(non_zero_index) + 1)], [])
            for each_index in subset_index:
                ans.append(self.mask_by_index_1to0(each_index, father_mask=father_mask))
            assert len(ans) == 2 ** torch.sum(father_mask)
        else:  # bit opertation
            shape = list(father_mask.shape)
            shape[0] = T_sample_limit
            ans = (torch.rand(shape) > 0.5).to(torch.int8)
            ans = ((ans + father_mask.repeat(T_sample_limit, 1)) > 1.5).to(torch.int8)  # bitwise and
        return ans

    def mask_by_complementary_index_1to0(self, index, father_mask=None):
        if father_mask is None:
            mask = copy.deepcopy(self._one_tensor_mask_flatten)
        else:
            mask = copy.deepcopy(father_mask)
        # print('index', index)
        for i in range(len(mask)):
            if i not in index:
                mask[i] = 0
        return mask

    def mask_by_index_0to1(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.zeros_like(self._zero_tensor_mask_flatten)
        else:
            mask = copy.deepcopy(father_mask)
        # print('mask_by_index_0to1 before mask', mask, father_mask)
        # print(index)
        for i in index:
            # print('mask_by_index_1to0 i', i)
            mask[0][i] = 1
        # print('mask_by_index_0to1 after mask', mask, father_mask)
        return mask.to(torch.int8)

    def supermask_by_index_0to1(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.zeros(1, len(self.super_index_to_index))
        else:
            mask = copy.deepcopy(father_mask)
        # print('mask_by_index_0to1 before mask', mask, father_mask)
        # print(index)
        for i in index:
            # print('mask_by_index_1to0 i', i)
            mask[0][i] = 1
        # print('mask_by_index_0to1 after mask', mask, father_mask)
        return mask.to(torch.int8)

    def supermask_to_mask(self, supermask):
        if torch.sum(supermask) > self.multiplyList(list(supermask.shape)):  # many 1
            mask = torch.ones_like(self.baseline_data_flatten)
            for i, each in enumerate(supermask[0]):
                if each == 0:
                    mask[0][self.super_index_to_index[i][0]: self.super_index_to_index[i][1]] = 0
        else:
            mask = torch.zeros_like(self.baseline_data_flatten)
            for i, each in enumerate(supermask[0]):
                if each == 1:
                    mask[0][self.super_index_to_index[i][0]: self.super_index_to_index[i][1]] = 1
        # print('supermask', supermask[0][0:20], torch.sum(supermask[0]))
        # print('mask', mask[0][0:20], torch.sum(mask[0]))
        return mask

    def mask_by_index_1to0(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.ones_like(self.template_data_flattern)
        else:
            mask = copy.deepcopy(father_mask)
        # print('mask_by_index_1to0 before mask', mask, father_mask)
        for i in index:
            # print('mask_by_index_1to0 i', i)
            mask[i] = 0
        # print('mask_by_index_1to0 after mask', mask, father_mask)
        return mask

    def mask_to_data(self, mask_flatten, data__flatten):
        return mask_flatten * data__flatten + (1 - mask_flatten) * self.template_data_flattern

    def return_T_S_pair(self, k, T_sample_limit=None):
        N_index = range(self.N)
        S_index_k_order_list = list(map(list, combinations(N_index, k)))
        # print('k, S_index_k_order_list', k, S_index_k_order_list)
        S_mask_flatten_list = []
        for each in S_index_k_order_list:
            temp_flatten_S = self.supermask_by_index_0to1(each)
            # print('temp_flatten_S', temp_flatten_S, temp_flatten_S.shape)
            assert torch.sum(temp_flatten_S) == k  # to be del
            S_mask_flatten_list.append(temp_flatten_S)
        # print('S_mask_flatten_list', S_mask_flatten_list)

        T_mask_flatten_list = []
        for each_S_mask in S_mask_flatten_list:
            temp_T_mask = self.return_all_permutation(torch.ones_like(each_S_mask, dtype=torch.int8) - each_S_mask,
                                                      T_sample_limit=T_sample_limit)
            T_mask_flatten_list.append(temp_T_mask)
        # print('T_mask_flatten_list', T_mask_flatten_list)

        return T_mask_flatten_list, S_mask_flatten_list

    def delta_S_F_T(self, S_mask, T_mask):
        sum = 0
        W_mask_list = self.return_all_permutation(S_mask)
        for W_mask in W_mask_list:
            super_mask = (W_mask + T_mask > 0.5).to(torch.int)  # set and
            # print(super_mask.shape)
            # print(self.sample_insulin_data_flatten.shape)
            # print(self.baseline_data_flatten.shape)
            mask = self.supermask_to_mask(super_mask)
            data_flatten = self.sample_insulin_data_flatten * mask + self.baseline_data_flatten * (1 - mask)
            data = self.convert_flatten_to_data(data_flatten)
            # super_mask = super_mask.reshape(1, -1)
            sum += (-1) ** (torch.sum(W_mask) - torch.sum(S_mask)) * self.F.insulin_forward(*data)
        return sum

    def stacked_delta_S_F_T(self, S_mask, T_mask):
        flatten_input_list = []
        w_list = []
        s_list = []
        W_mask_list = self.return_all_permutation(S_mask)
        for W_mask in W_mask_list:
            super_mask = (W_mask + T_mask > 0.5).to(torch.int)  # set and
            mask = self.supermask_to_mask(super_mask)
            data_flatten = self.sample_insulin_data_flatten * mask + self.baseline_data_flatten * (1 - mask)

            flatten_input_list.append(data_flatten)
            w_list.append(torch.sum(W_mask))
            s_list.append(torch.sum(S_mask))

        return flatten_input_list, w_list, s_list

    def calculate_shapley_taylor_stack(self, max_k):
        ans = []

        for k in range(1, max_k):
            flatten_input_list_stack = []
            w_list_stack = []
            s_list_stack = []
            t_count_for_each_s_stack = []
            ncr_stack = []
            T_mask_list, S_mask_list = self.return_T_S_pair(k, T_sample_limit=2)
            # print(len(T_mask_list))  # 183
            assert len(T_mask_list) == len(S_mask_list)
            for each_T, each_S in tqdm(zip(T_mask_list, S_mask_list)):
                summation = 0
                for each_each_T in each_T:
                    each_each_T = each_each_T.unsqueeze(0)
                    ncr = printNcR(self.N - 1, int(
                        torch.sum(each_each_T).numpy()))  # 183it [00:48,  3.74it/s]  183it [00:48,  3.78it/s]
                    flatten_input_list, w_list, s_list = self.stacked_delta_S_F_T(each_S, each_each_T)
                    flatten_input_list_stack += flatten_input_list
                    w_list_stack += w_list
                    s_list_stack += s_list
                    ncr_stack += [ncr] * len(w_list)
                t_count_for_each_s_stack.append(len(each_T))
                # _r = self.N - int(torch.sum(each_S).numpy())
                # _r = 2**(_r)

                # _res = _res * _r
                # _res = _res / ncr
            # assert len(flatten_input_list_stack)==len(w_list_stack)==len(s_list_stack)==len(ncr_stack)
            print(torch.vstack(flatten_input_list_stack).shape)
            flatten_input_list_stack_dataset = TensorDataset(torch.vstack(flatten_input_list_stack).to(self.device))
            flatten_input_list_stack_dataset_loader = DataLoader(flatten_input_list_stack_dataset, batch_size=512)
            out_list = []
            for batch_flatten_data in flatten_input_list_stack_dataset_loader:
                batch_flatten_data = batch_flatten_data[0]
                print(batch_flatten_data.shape)
                batch_data = self.convert_flatten_to_data(batch_flatten_data)
                out = self.F.insulin_forward(*batch_data)
                out_list.append(out.detach().cpu().numpy())
            out_list = np.vstack(out_list)
            w_list_stack = np.array(w_list_stack)
            s_list_stack = np.array(s_list_stack)
            ncr_stack = np.array(ncr_stack)
            # print(out_list.shape)  # (732, 7)
            # print(w_list_stack.shape)  # (732,)
            # print(s_list_stack.shape)  # (732,)
            # print(ncr_stack.shape)  # (732,)
            tmp = (-1) ** np.abs(w_list_stack - s_list_stack).reshape(-1, 1)
            batch_delta_S_F_T = tmp * out_list
            _r = self.N - s_list_stack
            _r = [2 ** int(each) for each in _r]
            _r = np.array(_r)
            _r = _r / ncr_stack
            batch_delta_S_F_T = batch_delta_S_F_T * _r.reshape(-1, 1)
            splits = np.cumsum([0] + t_count_for_each_s_stack)
            ans = []
            for _start, _stop in zip(splits[:-1], splits[1:]):
                ans.append(np.sum(batch_delta_S_F_T[_start: _stop], axis=0))
            print(len(ans))
            print(ans[0:5])

            # print(ans)
            # assert 1==0

        return torch.Tensor(ans)

    def calculate_shapley_taylor(self, max_k):
        ans = []
        for k in range(1, max_k):
            T_mask_list, S_mask_list = self.return_T_S_pair(k)
            for each_T, each_S in tqdm(zip(T_mask_list, S_mask_list)):
                summation = 0
                for each_each_T in each_T:
                    each_each_T = each_each_T.unsqueeze(0)
                    # print('====C', self.N-1, int(torch.sum(each_each_T).numpy()))
                    # summation += self.delta_S_F_T(each_S, each_each_T) / C(self.N-1, int(torch.sum(each_each_T).numpy()), method='dc')
                    ncr = printNcR(self.N - 1, int(torch.sum(each_each_T).numpy()))
                    _res = self.delta_S_F_T(each_S, each_each_T).to(torch.float64)
                    _r = self.N - int(torch.sum(each_S).numpy())
                    _r = 2 ** (_r)
                    # print(_r, type(_r))
                    # print(_res, type(_res))
                    # print(ncr, type(ncr))
                    # _res = _res * _r
                    # _res = _res / ncr
                    summation += _res

                I_S_k_F = k / self.N * summation
                ans.append(I_S_k_F.data)
                # print('k', k, each_S, I_S_k_F.data)
        print('len(ans)', len(ans))  # 183it [41:07, 13.48s/it]
        assert 1 == 0
        return torch.Tensor(ans)


class GroupKernelShapleyTaylor:
    def __init__(self, F, sample_input_tuple, examination_list=None, patient_limit=None, output_txt=None, save_dir=None):
        self.examination_list = examination_list
        sample_sugar_data, sample_insulin_data, y_sugar, y_insulin = sample_input_tuple
        self.patient_limit = patient_limit
        self.save_dir = save_dir
        if self.save_dir is not None:
            np.save(f'./{self.save_dir}/sample_input_tuple{self.patient_limit}.npy', sample_insulin_data)
        if output_txt is not None:
            self.f_translate = output_txt
        else:
            self.f_translate = None
        print('Warning, label sugar', y_sugar, 'label insulin', y_insulin)
        self.y_insulin = y_insulin
        sample_sugar_data, sample_insulin_data = list(sample_sugar_data), list(sample_insulin_data)
        # sample_insulin_data[3][0, 7] = 12  # 人工修改某个血糖
        # sample_insulin_data[3][0, 7] = 12  # 人工修改某个血糖
        print('血糖', sample_insulin_data[3].reshape(-1))
        # sample_insulin_data[1][0, 11, -1] = 4  # 人工修改某个胰岛素值
        self.sample_insulin_data = sample_insulin_data
        self.device = next(F.parameters()).device
        self.F = F.to(self.device)

        self._zero_tensor_mask = None
        self._one_tensor_mask = None
        self.size_list = None
        self.super_index_to_index = {}
        # init self._zero_tensor_mask, self.size_list do not modify
        self.init_group_mask(sample_insulin_data)
        self._zero_tensor_mask_flatten = self.convert_data_to_flatten(self._zero_tensor_mask)
        self._one_tensor_mask_flatten = self.convert_data_to_flatten(self._one_tensor_mask)

        self.sample_insulin_data_flatten = self.convert_data_to_flatten(sample_insulin_data).to(self.device)
        print('self.sample_insulin_data_flatten.shape', self.sample_insulin_data_flatten.shape)
        self.get_query_insulin_flattern(self.sample_insulin_data_flatten)
        self.get_time_data_flattern(self.sample_insulin_data_flatten)
        # sample_insulin_data = self.convert_flatten_to_data(sample_insulin_data_flatten)

        # self.n = len(self.N_flatten)
        # self.group_mask = None

    def create_list_of_zero_tensor(self, list_of_tensor_size, batch=0):
        res = []
        if batch == 0:
            for each in list_of_tensor_size:
                res.append(torch.zeros(each))
            raise NotImplementedError
        else:
            for each in list_of_tensor_size:
                if type(each) == list and len(each) > 0:
                    res.append(torch.zeros(each[1:]).unsqueeze(dim=0).repeat(batch, 1, 1))
                elif type(each) == list and len(each) == 0:
                    pass
                else:
                    raise NotImplementedError
        # for each in res:
        #     print('create_list_of_zero_tensor', each.shape)
        return res

    def create_list_of_one_tensor(self, list_of_tensor_size, batch=0):
        res = []
        if batch == 0:
            for each in list_of_tensor_size:
                res.append(torch.ones(each))
            raise NotImplementedError
        else:
            for each in list_of_tensor_size:
                if type(each) == list and len(each) > 0:
                    res.append(torch.ones(each[1:]).unsqueeze(dim=0).repeat(batch, 1, 1))
                elif type(each) == list and len(each) == 0:
                    pass
                else:
                    raise NotImplementedError
        # for each in res:
        #     print('create_list_of_one_tensor', each.shape)
        return res

    def multiplyList(self, myList):
        result = 1
        for x in myList:
            result = result * x
        return result

    def init_group_mask(self, data):
        if type(data) == tuple or type(data) == list:
            self.size_list = []
            self.size_list_product = []
            # self.group_mask = []
            # init self._zero_tensor_mask
            for each in data:
                if torch.is_tensor(each):
                    _shape = list(each.shape)
                    _shape[0] = 1
                    self.size_list.append(_shape)
                    self.size_list_product.append(self.multiplyList(_shape))
                elif type(each) == list and len(each) == 0:
                    self.size_list.append([])
                    self.size_list_product.append(0)
                else:
                    print(each)
                    raise NotImplementedError
            self._zero_tensor_mask = self.create_list_of_zero_tensor(self.size_list, batch=1)
            self._one_tensor_mask = self.create_list_of_one_tensor(self.size_list, batch=1)
            # init N & super_index_to_index
            _super_index_counter = 0
            _index_counter = 0
            self.super_index_to_name = {}
            _name_to_name = {1: 'insulin',
                             2: 'temp_insulin',
                             3: 'sugar',
                             4: 'drug',
                             5: 'days',
                             6: 'mask'}

            '''
            torch.Size([1, 1, 78])
            torch.Size([1, 21, 9])
            torch.Size([1, 21, 9])
            torch.Size([1, 21, 1])
            torch.Size([1, 21, 28])
            torch.Size([1, 21, 2])
            '''

            for _name, each in enumerate(self._zero_tensor_mask):
                if torch.is_tensor(each):
                    if each.shape[1] == 1:  # examination dim
                        self.examination_size = each.shape[2]
                        # print(each.shape[2])
                        for _i in range(each.shape[2]):
                            self.super_index_to_index[_super_index_counter] = (_index_counter, _index_counter + 1)
                            self.super_index_to_name[_super_index_counter] = self.examination_list[_super_index_counter]
                            _super_index_counter += 1
                            _index_counter += 1
                    else:
                        self.day_length = each.shape[1] / 7
                        # print(each.shape[1])
                        if each.shape[2] == 9:  # 是胰岛素维度
                            for _i in range(each.shape[1]):
                                _insulin_data = each[:, _i, :]  # torch.Size([1, 9])
                                self.super_index_to_index[_super_index_counter] = (
                                _index_counter, _index_counter + each.shape[2])
                                self.super_index_to_name[_super_index_counter] = f'{_name_to_name[_name]}_day{int(_i / 7)}_time{_i % 7}'
                                # self.super_index_to_name[_super_index_counter] = f'{_name_to_name[_name]}_day{int(_i / 7)}_time{_i % 7}_insulin{_insulin_data[0, 0].numpy().astype(int)}'
                                _super_index_counter += 1
                                _index_counter += each.shape[2]
                        elif each.shape[2] == 28:  # 是口服药的维度
                            for _i in range(each.shape[1]):
                                for _drug_index in range(each.shape[2]):
                                    self.super_index_to_index[_super_index_counter] = (
                                    _index_counter, _index_counter + 1)
                                    self.super_index_to_name[
                                        _super_index_counter] = f'{_name_to_name[_name]}_day{int(_i / 7)}_time{_i % 7}_drug{_drug_index}'
                                    _super_index_counter += 1
                                    _index_counter += 1
                        else:
                            for _i in range(each.shape[1]):
                                self.super_index_to_index[_super_index_counter] = (
                                _index_counter, _index_counter + each.shape[2])
                                self.super_index_to_name[
                                    _super_index_counter] = f'{_name_to_name[_name]}_day{int(_i / 7)}_time{_i % 7}'
                                _super_index_counter += 1
                                _index_counter += each.shape[2]
                elif type(each) == list and len(each) == 0:
                    pass
                else:
                    raise ValueError
            # np.save('tmp_super_index_to_name', self.super_index_to_name)
            # np.save('tmp_super_index_to_index', self.super_index_to_index)
            # print('self.super_index_to_name', self.super_index_to_name)
            # print('self.super_index_to_index', self.super_index_to_index)
            # assert 1==0
        else:
            print(data)
            raise NotImplementedError

    def convert_data_to_flatten(self, data):
        if type(data) == tuple or type(data) == list:
            if type(data) == tuple:
                data = list(data)
            for _i in range(len(data)):
                if torch.is_tensor(data[_i]):
                    # print(data[_i].shape[0])
                    data[_i] = data[_i].reshape(data[_i].shape[0], -1)
                elif type(data[_i]) == list and len(data[_i]) == 0:
                    data.remove([])  # 如果不是最后一个是list可能会有问题
                else:
                    print(data[_i])
                    print(type(data[_i]))
                    raise NotImplementedError
            # assert 1==0
            data = torch.hstack(data)
        elif type(data) == int:
            print(data)
            raise NotImplementedError
        elif type(data) == torch.Tensor:
            raise NotImplementedError
        else:
            print(data)
            raise NotImplementedError
        return data

    def size_splits(self, tensor, split_size_product, split_sizes, dim=0):
        """Splits the tensor according to chunks of split_sizes.

        Arguments:
            tensor (Tensor): tensor to split.
            split_size_product: product of size
            split_sizes (list(int)): sizes of chunks
            dim (int): dimension along which to split the tensor.
        """
        _batch = tensor.shape[0]
        if dim < 0:
            dim += tensor.dim()

        dim_size = tensor.size(dim)
        if dim_size != torch.sum(torch.Tensor(split_size_product)):
            raise KeyError("Sum of split sizes exceeds tensor dim")

        splits = torch.cumsum(torch.Tensor([0] + split_size_product), dim=0)[:-1]
        ans = []
        for start, length, shape in zip(splits, split_size_product, split_sizes):
            if len(shape) == 0:
                ans.append([])
            else:
                shape[0] = _batch
                ans.append(tensor.narrow(int(dim), int(start), int(length)).reshape(shape))

        return ans

    def convert_flatten_to_data(self, flatten_tensor):
        ans = self.size_splits(flatten_tensor, self.size_list_product, self.size_list, dim=1)
        return ans

    def assign_baseline_data(self, data, ignore_yesterday_missing=False, ignore_yesterday_exam_missing=False):
        self.baseline_data = data
        self.baseline_data_flatten = self.convert_data_to_flatten(data).to(self.device)

        # self.sample_insulin_data_flatten_narrow = self.convert_sparse_data_to_narrow_data(self.sample_insulin_data_flatten)
        # self.baseline_data_flatten_narrow = self.convert_sparse_baseline_to_narrow_baseline()

        self.sample_insulin_data_flatten_narrow, self.baseline_data_flatten_narrow = self.convert_sparse_to_narrow_union(
            self.sample_insulin_data_flatten, self.baseline_data_flatten, ignore_yesterday_missing=ignore_yesterday_missing,
            ignore_yesterday_exam_missing=ignore_yesterday_exam_missing)

        # print('aaa', self.sample_insulin_data_flatten[0, 78:78 + 21 * 9].reshape(21, 9))  # insulin
        self.target_insulin_time_list = []
        print("Warning, unstable performance")
        for _i, _each in enumerate(self.sample_insulin_data_flatten[0, 78:78 + 21 * 9].reshape(21, 9)[-7:, :]):
            if _each[0] != -1 and _each[-1] == -1:
                self.target_insulin_time_list.append(_i)
        if len(self.target_insulin_time_list) == 0:
            return "No target insulin, continue"

        self.target_insulin_type_list = []
        print("Warning, unstable performance")
        for _i, _each in enumerate(self.sample_insulin_data_flatten[0, 78:78 + 21 * 9].reshape(21, 9)[-7:, :]):
            if _each[0] != -1 and _each[-1] == -1:
                self.target_insulin_type_list.append(_each[0])
        if len(self.target_insulin_type_list) == 0:
            return "No target insulin, continue"

        # print('self.sample_insulin_data_flatten_narrow', self.sample_insulin_data_flatten_narrow)

        self.super_index_to_index_narrow = {}  # 是self.super_index_to_index的压缩版本，value是连续的，最大是压缩后向量维度
        _pointer = 0
        _index_helper = 0
        for i in range(len(self.super_index_to_index)):
            if self.super_index_narrowed[i]:  # this dim is narrowed
                pass
            else:
                self.super_index_to_index_narrow[_pointer] = (_index_helper, _index_helper + self.super_index_to_index[i][1]-self.super_index_to_index[i][0])
                _index_helper += self.super_index_to_index[i][1]-self.super_index_to_index[i][0]
                _pointer += 1

        # 检验用
        _counter = 0
        for i in range(len(self.super_index_narrowed)):
            if not self.super_index_narrowed[i]:
                _counter += 1
        assert len(self.super_index_to_index_narrow) == _counter

        self.narrow_index_to_normal_index = {}  # 压缩后的index到压缩前的index的映射，用来最后从压缩shap系数矩阵复原未压缩稀疏矩阵
        _counter = 0
        for i in range(len(self.super_index_to_index)):
            this_dim_narrowed = self.super_index_narrowed[i]
            if this_dim_narrowed:
                pass
            else:
                self.narrow_index_to_normal_index[_counter] = i
                _counter += 1
        assert len(self.narrow_index_to_normal_index) == len(self.super_index_to_index_narrow)
        self.normal_index_to_narrow_index = {value: key for key, value in self.narrow_index_to_normal_index.items()}


        self.narrow_index_to_class_name = {}  # 压缩后的index到类别名称的映射, 取自解释内容的名字一致。
        _counter = 0
        for i in range(len(self.narrow_index_to_normal_index)):
            self.narrow_index_to_class_name[i] = self.super_index_to_name[self.narrow_index_to_normal_index[i]]

        # narrow bound code block, expired
        # self.narrow_bound = {}
        # for narrow_index in range(len(self.super_index_to_index_narrow)):
        #     original_index = self.narrow_index_to_normal_index[narrow_index]
        #     original_index_start = self.super_index_to_index[original_index][0]
        #     original_index_end = self.super_index_to_index[original_index][1]
        #     original_data = self.sample_insulin_data_flatten[0, original_index_start: original_index_end]
        #     if original_index_end - original_index_start == 1 and original_index < 80:  # examination
        #         self.narrow_bound[narrow_index] = (-np.inf, np.inf)
        #     elif original_index_end - original_index_start == 9:  # insulin
        #         if original_data[-1]==-1:  # 种类胰岛素，没有值
        #             self.narrow_bound[narrow_index] = (-np.inf, np.inf)
        #         else:  # 胰岛素有值
        #             self.narrow_bound[narrow_index] = (-np.inf, np.inf)
        #     elif original_index_end - original_index_start == 1:  # sugar
        #         if original_data >7:
        #             self.narrow_bound[narrow_index] = (0, np.inf)
        #         else:
        #             self.narrow_bound[narrow_index] = (-np.inf, 0)
        #     elif original_index_end - original_index_start == 28:  # drug
        #         self.narrow_bound[narrow_index] = (-np.inf, 0)
        #     elif original_index_end - original_index_start == 2:  # day
        #         self.narrow_bound[narrow_index] = (-np.inf, np.inf)
        #     else:
        #         raise ValueError
        # # print(self.narrow_bound)
        # assert len(self.narrow_bound) == len(self.super_index_to_index_narrow)

    def return_all_permutation(self, father_mask, T_sample_limit=None):
        if T_sample_limit is None:  # slow algorithm, return all permutation
            ans = []
            non_zero_index = np.where(father_mask == 1)[0]
            subset_index = sum(
                [list(map(list, combinations(non_zero_index, i))) for i in range(len(non_zero_index) + 1)], [])
            for each_index in subset_index:
                ans.append(self.mask_by_index_1to0(each_index, father_mask=father_mask))
            assert len(ans) == 2 ** torch.sum(father_mask)
        else:  # bit opertation
            shape = list(father_mask.shape)
            shape[0] = T_sample_limit
            ans = (torch.rand(shape) > 0.5).to(torch.int8)
            ans = ((ans + father_mask.repeat(T_sample_limit, 1)) > 1.5).to(torch.int8)  # bitwise and
        return ans

    def mask_by_complementary_index_1to0(self, index, father_mask=None):
        if father_mask is None:
            mask = copy.deepcopy(self._one_tensor_mask_flatten)
        else:
            mask = copy.deepcopy(father_mask)
        # print('index', index)
        for i in range(len(mask)):
            if i not in index:
                mask[i] = 0
        return mask

    def mask_by_index_0to1(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.zeros_like(self._zero_tensor_mask_flatten)
        else:
            mask = copy.deepcopy(father_mask)
        # print('mask_by_index_0to1 before mask', mask, father_mask)
        # print(index)
        for i in index:
            # print('mask_by_index_1to0 i', i)
            mask[0][i] = 1
        # print('mask_by_index_0to1 after mask', mask, father_mask)
        return mask.to(torch.int8)

    def supermask_by_index_0to1(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.zeros(1, len(self.super_index_to_index))
        else:
            mask = copy.deepcopy(father_mask)
        # print('mask_by_index_0to1 before mask', mask, father_mask)
        # print(index)
        for i in index:
            # print('mask_by_index_1to0 i', i)
            mask[0][i] = 1
        # print('mask_by_index_0to1 after mask', mask, father_mask)
        return mask.to(torch.int8)

    def supermask_to_mask_batch(self, supermask, use_narrow=False):
        if not use_narrow:
            assert supermask.shape[1] == len(
                self.super_index_to_index), f"{supermask.shape}, {len(self.super_index_to_index)}"
            mask = []
            for each_col_index in range(supermask.shape[1]):
                mask_col_num_tuple = self.super_index_to_index[each_col_index]
                mask_col_num = mask_col_num_tuple[1] - mask_col_num_tuple[0]
                supermask_col = supermask[:, each_col_index].unsqueeze(1).expand(-1, mask_col_num)
                mask.append(supermask_col)
            mask = torch.hstack(mask)
        else:
            assert supermask.shape[1] == len(
                self.super_index_to_index_narrow), f"{supermask.shape}, {len(self.super_index_to_index_narrow)}"
            mask = []
            for each_col_index in range(supermask.shape[1]):
                mask_col_num_tuple = self.super_index_to_index_narrow[each_col_index]
                mask_col_num = mask_col_num_tuple[1] - mask_col_num_tuple[0]
                supermask_col = supermask[:, each_col_index].unsqueeze(1).expand(-1, mask_col_num)
                mask.append(supermask_col)
            mask = torch.hstack(mask)

        return mask

    def mask_by_index_1to0(self, index, father_mask=None):
        if father_mask is None:
            mask = torch.ones_like(self.template_data_flattern)
        else:
            mask = copy.deepcopy(father_mask)
        # print('mask_by_index_1to0 before mask', mask, father_mask)
        for i in index:
            # print('mask_by_index_1to0 i', i)
            mask[i] = 0
        # print('mask_by_index_1to0 after mask', mask, father_mask)
        return mask

    def mask_to_data(self, mask_flatten, data__flatten):
        return mask_flatten * data__flatten + (1 - mask_flatten) * self.template_data_flattern

    def st_kernel(self, samples):
        s = samples.sum(axis=1)
        M = np.ones(samples.shape[0]) * samples.shape[1]
        return (M - 1) / (comb(M, s) * comb(s, 2) * (M - s))

    def shap_sampler(self, d1, n_samples=1000):
        inf_samples = np.concatenate([
            np.ones((1, d1)),
            np.zeros((1, d1)),
            # np.eye(d1 + d2)
        ], axis=0)
        reg_samples = (np.random.rand(n_samples - inf_samples.shape[0], d1) > .5)

        reg_weights = self.st_kernel(reg_samples)
        reg_weights /= reg_weights.sum()
        reg_weights *= reg_weights.shape[0]
        inf_weights = np.ones(inf_samples.shape[0])
        inf_weights /= inf_weights.shape[0]
        inf_weights *= reg_weights.shape[0] * 3

        weights = np.concatenate([reg_weights, inf_weights], axis=0)
        samples = np.concatenate([reg_samples, inf_samples], axis=0)
        return torch.from_numpy(samples[:, :d1]), \
               torch.from_numpy(weights)

    def get_query_insulin_flattern(self, data):
        assert data.shape[0] == 1, 'only support batch=1'
        self.query_insulin_data_flatten = torch.zeros(self.sample_insulin_data_flatten.shape).to(self.sample_insulin_data_flatten.device)
        for id in range(len(self.super_index_to_index)):
            if self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 9:  # insulin部分
                if data[0, self.super_index_to_index[id][0]: self.super_index_to_index[id][1]][-1] == -1:  # 最后一个是-1 胰岛素 insulin
                    if data[0, self.super_index_to_index[id][0]: self.super_index_to_index[id][1]][0] != -1:  # 第一个不是-1 胰岛素 insulin
                        self.query_insulin_data_flatten[0, self.super_index_to_index[id][0]: self.super_index_to_index[id][1]]\
                            = data[0, self.super_index_to_index[id][0]: self.super_index_to_index[id][1]] + 1  # 为了抵消掉 -1 的影响，全都要加1

    def get_time_data_flattern(self, data):  # 获取时间编码
        assert data.shape[0] == 1, 'only support batch=1'
        self.time_sequence_data_flatten = torch.zeros(self.sample_insulin_data_flatten.shape).to(self.sample_insulin_data_flatten.device)
        for id in range(len(self.super_index_to_index)):
            if self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 2:  # time部分
                self.time_sequence_data_flatten[0, self.super_index_to_index[id][0]: self.super_index_to_index[id][1]]\
                    = data[0, self.super_index_to_index[id][0]: self.super_index_to_index[id][1]] + 1  # 为了抵消掉 -1 的影响，全都要加1
        np.save('time_sequence_data_flatten', self.time_sequence_data_flatten.cpu().numpy())

    def convert_sparse_data_to_narrow_data(self, data):
        assert data.shape[0]==1, 'only support batch=1'
        self.super_index_narrowed = {}  # 维度是否被narrow了，key为未压缩的数量，value为True/False

        narrow_data = []
        for each_data in data:  # data in batch (batch=1)
            each_narrow_data = []
            for id in range(len(self.super_index_to_index)):
                name_of_id = self.super_index_to_name[id]
                if self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' not in name_of_id:  # 是examination的部分
                    if each_data[self.super_index_to_index[id][0]] != -1:  # has value
                        each_narrow_data.append(each_data[self.super_index_to_index[id][0]])
                        self.super_index_narrowed[id] = False
                    else:
                        self.super_index_narrowed[id] = True
                elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' in name_of_id:  # 是drug的部分
                    if torch.sum(each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]])==0:  # do not have value
                        self.super_index_narrowed[id] = True
                    else:
                        each_narrow_data.append(each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]])
                        self.super_index_narrowed[id] = False
                elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 2:  # 是day的部分
                    '''
                    全面删除day带来的影响，如果不需要这样做可以直接注释这一段elif
                    '''
                    self.super_index_narrowed[id] = True
                else:  # 是其他的部分
                    if torch.sum(each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]])==-1*(self.super_index_to_index[id][1]-self.super_index_to_index[id][0]):  # 全是-1, do not have value
                        # do not have value
                        self.super_index_narrowed[id] = True
                    elif each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]][-1] == -1:  # 最后一个是-1 胰岛素 insulin
                        self.super_index_narrowed[id] = True
                        # 作为query的胰岛素，最后要补回来
                    else:
                        each_narrow_data.append(each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]])
                        self.super_index_narrowed[id] = False
            each_narrow_data = torch.hstack(each_narrow_data)
            narrow_data.append(each_narrow_data)

        narrow_data = torch.hstack(narrow_data)
        narrow_data = narrow_data.reshape(1, -1)
        # print(narrow_data.shape)
        return narrow_data


    def convert_sparse_to_narrow_union(self, data, baseline, ignore_yesterday_missing=False, ignore_yesterday_exam_missing=False):
        assert data.shape[0]==1, 'only support batch=1'
        assert baseline.shape[0]==1, 'only support batch=1'
        self.super_index_narrowed = {}  # 维度是否被narrow了，key为未压缩的数量，value为True/False
        self.super_index_narrowed_dataonly = {}  # 维度是否被narrow了，key为未压缩的数量，value为True/False
        self.super_index_narrowed_baselineonly = {}  # 维度是否被narrow了，key为未压缩的数量，value为True/False



        # denoted as data and baseline

        # print(self.super_index_to_index)  # 上面三个的key都是0~749，value是0~1107

        for each_data in data:  # data in batch (batch=1)  # each_data.shape torch.Size([1107])
            for id in range(len(self.super_index_to_index)):
                name_of_id = self.super_index_to_name[id]
                if self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' not in name_of_id:  # 是examination的部分
                    if each_data[self.super_index_to_index[id][0]] != -1:  # has value
                        self.super_index_narrowed_dataonly[id] = False
                    else:
                        self.super_index_narrowed_dataonly[id] = True
                elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' in name_of_id:  # 是drug的部分
                    if torch.sum(each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]])==0:  # do not have value
                        self.super_index_narrowed_dataonly[id] = True
                    else:
                        self.super_index_narrowed_dataonly[id] = False
                elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 2:  # 是day的部分
                    '''
                    全面删除day带来的影响，如果不需要这样做可以直接注释这一段elif
                    '''
                    self.super_index_narrowed_dataonly[id] = True
                else:  # 是其他的部分
                    if torch.sum(each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]])==-1*(self.super_index_to_index[id][1]-self.super_index_to_index[id][0]):  # 全是-1, do not have value
                        # do not have value
                        self.super_index_narrowed_dataonly[id] = True
                    elif each_data[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]][-1] == -1:  # 最后一个是-1 胰岛素 insulin
                        self.super_index_narrowed_dataonly[id] = True
                        # 作为query的胰岛素，最后要补回来
                    else:
                        self.super_index_narrowed_dataonly[id] = False
        if ignore_yesterday_missing:
            self.super_index_narrowed_baselineonly = copy.deepcopy(self.super_index_narrowed_dataonly)
        elif ignore_yesterday_exam_missing:
            for each_baseline, each_data in zip(baseline, data):  # data in batch (batch=1)
                for id in range(len(self.super_index_to_index)):
                    name_of_id = self.super_index_to_name[id]
                    if self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' not in name_of_id:  # 是examination的部分
                        if each_data[self.super_index_to_index[id][0]] != -1:  # has value  # 这里改变了
                            self.super_index_narrowed_baselineonly[id] = False
                        else:
                            self.super_index_narrowed_baselineonly[id] = True
                    elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' in name_of_id:  # 是drug的部分
                        if torch.sum(each_baseline[self.super_index_to_index[id][0]: self.super_index_to_index[id][
                            1]]) == 0:  # do not have value
                            self.super_index_narrowed_baselineonly[id] = True
                        else:
                            self.super_index_narrowed_baselineonly[id] = False
                    elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 2:  # 是day的部分
                        '''
                        全面删除day带来的影响，如果不需要这样做可以直接注释这一段elif
                        '''
                        self.super_index_narrowed_baselineonly[id] = True
                    else:  # 是其他的部分
                        if torch.sum(each_baseline[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]]) == -1 * (
                                self.super_index_to_index[id][1] - self.super_index_to_index[id][
                            0]):  # 全是-1, do not have value
                            # do not have value
                            self.super_index_narrowed_baselineonly[id] = True
                        elif each_baseline[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]][
                            -1] == -1:  # 最后一个是-1 胰岛素 insulin
                            self.super_index_narrowed_baselineonly[id] = True
                            # 作为query的胰岛素，最后要补回来
                        else:
                            self.super_index_narrowed_baselineonly[id] = False
        else:
            for each_baseline in baseline:  # data in batch (batch=1)
                for id in range(len(self.super_index_to_index)):
                    name_of_id = self.super_index_to_name[id]
                    if self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' not in name_of_id:  # 是examination的部分
                        if each_baseline[self.super_index_to_index[id][0]] != -1:  # has value
                            self.super_index_narrowed_baselineonly[id] = False
                        else:
                            self.super_index_narrowed_baselineonly[id] = True
                    elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 1 and 'drug_' in name_of_id:  # 是drug的部分
                        if torch.sum(each_baseline[self.super_index_to_index[id][0]: self.super_index_to_index[id][
                            1]]) == 0:  # do not have value
                            self.super_index_narrowed_baselineonly[id] = True
                        else:
                            self.super_index_narrowed_baselineonly[id] = False
                    elif self.super_index_to_index[id][1] - self.super_index_to_index[id][0] == 2:  # 是day的部分
                        '''
                        全面删除day带来的影响，如果不需要这样做可以直接注释这一段elif
                        '''
                        self.super_index_narrowed_baselineonly[id] = True
                    else:  # 是其他的部分
                        if torch.sum(each_baseline[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]]) == -1 * (
                                self.super_index_to_index[id][1] - self.super_index_to_index[id][
                            0]):  # 全是-1, do not have value
                            # do not have value
                            self.super_index_narrowed_baselineonly[id] = True
                        elif each_baseline[self.super_index_to_index[id][0]: self.super_index_to_index[id][1]][
                            -1] == -1:  # 最后一个是-1 胰岛素 insulin
                            self.super_index_narrowed_baselineonly[id] = True
                            # 作为query的胰岛素，最后要补回来
                        else:
                            self.super_index_narrowed_baselineonly[id] = False

        assert len(self.super_index_narrowed_dataonly)==len(self.super_index_narrowed_baselineonly)
        for i in range(len(self.super_index_narrowed_dataonly)):
            self.super_index_narrowed[i] = self.super_index_narrowed_dataonly[i] and self.super_index_narrowed_baselineonly[i]

        narrow_data = []
        narrow_baseline = []
        for id in range(len(self.super_index_narrowed)):
            this_id_narrowedornot = self.super_index_narrowed[id]
            if this_id_narrowedornot:  # diminished
                pass
            else:
                narrow_data.append(data[0][ self.super_index_to_index[id][0]: self.super_index_to_index[id][1] ])
                narrow_baseline.append(baseline[0][ self.super_index_to_index[id][0]: self.super_index_to_index[id][1] ])

        narrow_data = torch.hstack(narrow_data)
        narrow_baseline = torch.hstack(narrow_baseline)

        # print(self.super_index_narrowed)
        # print(self.super_index_narrowed_dataonly)
        # np.save('tmp_super_index_narrowed', self.super_index_narrowed)
        # np.save('tmp_super_index_narrowed_dataonly', self.super_index_narrowed_dataonly)
        # np.save('tmp_super_index_narrowed_baselineonly', self.super_index_narrowed_baselineonly)

        return narrow_data, narrow_baseline


    def convert_narrow_data_to_sparse_data(self, narrow_data):
        device = narrow_data.device
        assert narrow_data.shape[0]==1, 'only support batch=1'
        data = []
        _pointer = 0
        for id in range(len(self.super_index_narrowed)):
            this_dim_narrowed = self.super_index_narrowed[id]
            if this_dim_narrowed:  # should add -1 or 0
                if (self.super_index_to_index[id][1] - self.super_index_to_index[id][0])==28:
                    _tmp = torch.zeros(self.super_index_to_index[id][1] - self.super_index_to_index[id][0]).to(device).reshape(-1)
                    # print('_tmp', _tmp)
                    data.append(_tmp)
                else:
                    _tmp =-torch.ones(self.super_index_to_index[id][1] - self.super_index_to_index[id][0]).to(device).reshape(-1)
                    # print('_tmp', _tmp)
                    data.append(_tmp)
                # _pointer += self.super_index_to_index[id][1] - self.super_index_to_index[id][0]
            else:  # directly append
                data.append(narrow_data[0, _pointer: _pointer+self.super_index_to_index[id][1]-self.super_index_to_index[id][0]].reshape(-1))
                _pointer_add = self.super_index_to_index[id][1] - self.super_index_to_index[id][0]
                _pointer = _pointer + _pointer_add
            # if id ==5:
            #     assert 1==0
        data = torch.hstack(data)
        data = data.reshape(1, -1)
        return data

    def convert_sparse_baseline_to_narrow_baseline(self):
        assert len(self.super_index_narrowed)==len(self.super_index_to_index)
        device = self.baseline_data_flatten.device
        narrow_data = []
        for id in range(len(self.super_index_narrowed)):
            this_dim_narrowed = self.super_index_narrowed[id]
            if this_dim_narrowed:  # need narrow
                pass
            else:  # do not narrow
                narrow_data.append(self.baseline_data_flatten[0, self.super_index_to_index[id][0]: self.super_index_to_index[id][1]])
        narrow_data = torch.hstack(narrow_data)
        narrow_data = narrow_data.reshape(1, -1)
        return narrow_data

    def convert_sparse_supermask_to_narrow_supermask(self, supermask):
        print('supermask shape', supermask.shape)
        batch_size = supermask.shape[0]
        assert supermask.shape[1]==len(self.super_index_to_index)
        narrow_supermask = []
        for id in range(len(self.super_index_narrowed)):
            this_dim_narrowed = self.super_index_narrowed[id]
            if this_dim_narrowed:  # need narrow
                pass
            else:  # do not narrow
                narrow_supermask.append(supermask[:, id])
        narrow_supermask = torch.hstack(narrow_supermask)
        narrow_supermask = narrow_supermask.reshape(batch_size, -1)
        # print(narrow_supermask.shape, len(self.super_index_to_index_narrow))  # torch.Size([17000, 47]) 47
        return narrow_supermask

    def convert_narrow_supermask_to_supermask(self, narrow_supermask):
        batch_size = narrow_supermask.shape[0]
        assert narrow_supermask.shape[1]==len(self.super_index_to_index_narrow)
        supermask = torch.zeros(batch_size, len(self.super_index_to_index))
        for _i in range(narrow_supermask.shape[0]):
            for _j in range(narrow_supermask.shape[1]):
                supermask[_i, self.narrow_index_to_normal_index[_j]] = narrow_supermask[_i, _j]
        return supermask


    def fit(self, examination_list=None, use_narrow=False):
        torch.manual_seed(0)
        np.random.seed(0)

        if len(self.super_index_to_index_narrow) > 150:
            print("Error dimension is too high", len(self.super_index_to_index_narrow))
            time.sleep(2)
            return 0

        if use_narrow:
            n_samples = int((len(self.super_index_to_index_narrow) + 1) * len(self.super_index_to_index_narrow) / 2 * 3)  # (1+n)*n/2 * N
            with torch.no_grad():
                random_supermask_narrow, weights = self.shap_sampler(len(self.super_index_to_index_narrow), n_samples=n_samples)
                if np.isnan(weights).sum() > 0:
                    print("Nan weights, failed to explain")
                    return "Nan weights, failed to explain"

                random_supermask = self.convert_narrow_supermask_to_supermask(random_supermask_narrow)
                random_mask = self.supermask_to_mask_batch(random_supermask, use_narrow=False).to(self.device)
                random_mask_narrow = self.supermask_to_mask_batch(random_supermask_narrow, use_narrow=True).to(self.device)

                print('random_supermask_narrow.shape, random_mask_narrow.shape',
                      random_supermask_narrow.shape, random_mask_narrow.shape)  # torch.Size([n_samples, 49]) torch.Size([n_samples, 113])
                print('random_supermask.shape, random_mask.shape',
                      random_supermask.shape, random_mask.shape)  # torch.Size([n_samples, 750]) torch.Size([n_samples, 1107])
                values = []

                _input_flatten = random_mask * self.sample_insulin_data_flatten + (
                        1 - random_mask) * self.baseline_data_flatten

                _dataset = TensorDataset(_input_flatten.to(torch.float))
                _data_loader = DataLoader(_dataset, batch_size=512, shuffle=False)
                for _data in tqdm(_data_loader):
                    _data = self.convert_flatten_to_data(_data[0] + self.query_insulin_data_flatten + self.time_sequence_data_flatten)
                    values.append(self.F.insulin_forward(*_data).detach().cpu())
                values = torch.cat(values).squeeze().numpy()  # (17000, 7)
                print('Warning all original and all absent values ', values[-2:, :])
                values = values[:, np.array(self.target_insulin_time_list)]  # 只用目标insulin点做拟合

                d1 = len(self.super_index_to_index_narrow)  # 48: (112, 113)

                print("making features")
                dim = d1
                samples = random_supermask
                print('dim', dim)  # 49
                cross_terms = np.apply_along_axis(lambda s: (np.outer(s, s)).reshape(-1), 1, samples)  # 每个sample自己与自己做外积
                print('cross_terms', cross_terms.shape)  # cross_terms (n_sample, normal index num^2)
                term_supermask_narrow = np.triu(np.ones((dim, dim), dtype=bool), k=0)
                term_supermask = np.zeros((random_supermask.shape[1], random_supermask.shape[1]), dtype=bool)

                for _i in range(term_supermask_narrow.shape[0]):
                    for _j in range(term_supermask_narrow.shape[1]):
                        if term_supermask_narrow[_i, _j] == 1:
                            term_supermask[self.narrow_index_to_normal_index[_i], self.narrow_index_to_normal_index[_j]] = 1
                assert np.sum(term_supermask_narrow) == np.sum(term_supermask)
                terms_to_keep = np.where(term_supermask.reshape(-1))  # supermask to keep
                print('terms_to_keep', len(terms_to_keep), terms_to_keep[0].shape)  # terms_to_keep 1 (上三角矩阵的元素个数)
                _tmp = cross_terms[:, terms_to_keep]
                print('_tmp', _tmp.shape)  # (600, 1, 300)
                cross_terms = _tmp.squeeze()
                print('_tmp', cross_terms.shape)  # (600, 300)
                features = cross_terms.astype(bool)
                weights = weights * (1 / weights.max())

                print("fitting model")
                _start_time = time.time()

            # start to record gradient

            model = GradientDescendRidge(self.sample_insulin_data, self.baseline_data,
                                         self.sample_insulin_data_flatten_narrow, self.baseline_data_flatten_narrow,
                                         w_l2=0, w_l1=0, fit_intercept=True, lr=0.05, init_method='analytic',
                                         narrow_index_to_name=self.narrow_index_to_class_name,
                                         super_index_to_name=self.super_index_to_name,
                                         super_index_narrowed_baselineonly=self.super_index_narrowed_baselineonly,
                                         super_index_narrowed_dataonly=self.super_index_narrowed_dataonly,
                                         narrow_index_to_normal_index=self.narrow_index_to_normal_index,
                                         # insulin_l1=0.001,
                                         # insulin_l2=0.001,
                                         # exam_l1=0.001,
                                         # exam_l2=0.001,
                                         # sugar_l1=0.001,
                                         sugar_pos_neg=True,
                                         drug_pos_neg=True,
                                         mid_sugar_insulin_limit=True,
                                         sugar_high_range_insulin_pos=True,
                                         target_insulin_time_list = self.target_insulin_time_list,
                                         target_insulin_type_list = self.target_insulin_type_list,
                                         )  # analytic random
            final_epoch_output_info = model.fit(features, values, weights)
            print('features shape', features.shape)  # features shape (17000, 1653)
            print('values shape', values.shape)  # values shape (17000, 7)
            print('weights shape', weights.shape)  # weights shape torch.Size([17000])
            print("fit model, time", time.time() - _start_time)
            _predict_output = model.predict(torch.FloatTensor(features).cuda())  # torch.Size([17000, 7])
            print(_predict_output.cpu().numpy().shape, values.shape)
            _predict_residual = (_predict_output.cpu().numpy() - values) ** 2
            _predict_residual = np.mean(_predict_residual, axis=0)
            print('Warning fit residual mean square', _predict_residual)
            print('Warning fit residual mean absolute', np.mean(np.abs(_predict_output.cpu().numpy() - values)))
            mse_f = torch.nn.MSELoss()
            print('Warning fit residual mse', mse_f(_predict_output, torch.FloatTensor(values).cuda()))


            self.full_coeff = np.zeros((values.shape[1], len(self.super_index_to_index) * len(self.super_index_to_index)))  # (target time num, . .)
            self.full_coeff_narrow = np.zeros((values.shape[1], len(self.super_index_to_index_narrow) * len(self.super_index_to_index_narrow)))  # (target time num, . .)

            # assign value to full_coeff
            self.full_coeff[:, terms_to_keep] = model.coef_[:, :cross_terms.shape[1]].unsqueeze(1).cpu().numpy()
            self.full_coeff_narrow[:, np.where(np.triu(np.ones((len(self.super_index_to_index_narrow), len(self.super_index_to_index_narrow)))).reshape(-1)==1)] = model.coef_[:, :cross_terms.shape[1]].unsqueeze(1).cpu().numpy()
            self.full_coeff = self.full_coeff.reshape(values.shape[1], len(self.super_index_to_index), len(self.super_index_to_index))
            self.full_coeff_narrow = self.full_coeff_narrow.reshape(values.shape[1], len(self.super_index_to_index_narrow), len(self.super_index_to_index_narrow))
            self.weights = torch.from_numpy(self.full_coeff).to(torch.float32).cuda()
            print('self.full_coeff sum', self.full_coeff.sum(axis=(1, 2)))

            self.full_coeff_fullsize = np.zeros((7, len(self.super_index_to_index), len(self.super_index_to_index)))
            for _iii, _each_target_time in enumerate(self.target_insulin_time_list):
                self.full_coeff_fullsize[_each_target_time] = self.full_coeff[_iii]
            if self.save_dir is not None:
                np.save(f'./{self.save_dir}/full_coeff_{self.patient_limit}.npy', self.full_coeff_fullsize)

            print(self.narrow_index_to_normal_index)
            for each in self.narrow_index_to_normal_index:
                print(model.translation_sentence_list_full[self.narrow_index_to_normal_index[each]])
            # assert 1==0

            return model, self.full_coeff, values[-1], values[-2], _predict_residual, \
              self.target_insulin_time_list, self.y_insulin, self.f_translate, self.super_index_to_name, self.super_index_to_index, \
              self.sample_insulin_data_flatten, self.normal_index_to_narrow_index, final_epoch_output_info, self.narrow_index_to_normal_index
        else:
            assert 1==0  # 顺序不对，random mask

    def calculate_shapley_taylor(self, max_k=2, use_narrow=False):
        exam_encoder = ExaminationEncoder('not_use')
        examination_list = exam_encoder.examination_list
        return self.fit(examination_list=examination_list, use_narrow=use_narrow)

        # return torch.Tensor(ans)



class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x1 = x[:, 0].reshape(-1, 1)
        x2 = x[:, 1].reshape(-1, 1)
        x3 = x[:, 2].reshape(-1, 1)
        x = torch.hstack((x1, x2, x3))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x


if __name__ == "__main__":

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子


    def f_function(x):
        a = x[:, 0]
        b = x[:, 1]
        c = x[:, 2]
        # return 3*a+2*b+10*a*b+c
        return 1 * a + 4 * b + c


    x = np.random.random((1000, 3))
    y = f_function(x)

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    x_mean = x.mean(axis=0)
    x_zero = torch.zeros_like(x_mean)
    x_one = torch.ones_like(x_mean)
    x_random_small = torch.Tensor(np.random.random((1, 3)) * 0.1)
    x_group = x[200:300, :]

    network = SimpleNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=200)

    x_baseline = x_zero

    for _ in range(4000):
        ep_loss = 0
        network.train()
        for x, y in dataloader:
            y_pred = network(x)
            y = y.reshape(-1, 1)
            loss = F.mse_loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
        if _ % 100 == 0:
            network.eval()
            # print(ep_loss, y_pred[0:5].reshape(-1), y[0:5].reshape(-1))
            print(ep_loss)

            shap_values_list = []

            for _data_index in np.arange(200):
                # for _data_index in [0, 1, 3]:
                shapleytaylor = ShapleyTaylor(network, x[_data_index])
                shapleytaylor.assign_baseline_data(x_baseline)
                # print(shapleytaylor.template_data)
                shap_values = shapleytaylor.calculate_shapley_taylor(max_k=3)
                print('data', x[_data_index],
                      'baseline', network(x_baseline.reshape(1, -1)).data,
                      'pred', network(x[_data_index].reshape(1, -1)).data,
                      'label', y[_data_index])
                shap_values_list.append(shap_values)
            #     e = shap.DeepExplainer(network, x_baseline.reshape(1, -1))
            #     shap_values = e.shap_values(x[_data_index].reshape(1, -1))
            #     print('pred', network(x[_data_index].reshape(1, -1)).data,
            #           'baseline', network(x_baseline.reshape(1, -1)).data,
            #           'label', y[_data_index],
            #           'shap', shap_values,
            #           shap_values[0][0] / x[_data_index][0],
            #           shap_values[0][1] / x[_data_index][1],
            #           shap_values[0][2] / x[_data_index][2],
            #           )
            #     shap_values_list.append(shap_values[0])
            print(np.mean(np.vstack(shap_values_list), axis=0))

    # print(y_pred.shape)
    # print(y_pred[0:5])
    # print(y[0:5])
