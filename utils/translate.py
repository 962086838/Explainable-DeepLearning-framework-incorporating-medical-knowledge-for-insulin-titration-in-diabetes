import numpy as np

def translate(model, full_coeff, model_baseline_prediction, model_prediction, _predict_residual,
              target_insulin_time_list, y_insulin, f_translate, super_index_to_name, super_index_to_index,
              sample_insulin_data_flatten, normal_index_to_narrow_index, final_epoch_output_info):
    if f_translate is not None:
        _text_prefix_index = 0
        for _i, each in enumerate(full_coeff):  # each.shape (750, 750)
            print('Predict time', target_insulin_time_list[_i],
                  '基线剂量%.2f' % model_baseline_prediction[_i],
                  '模型预测剂量%.2f' % model_prediction[_i],
                  '实际剂量%.2f' % y_insulin[0, target_insulin_time_list[_i]].numpy(),
                  'fit residual%.2f' % _predict_residual[_i],
                  file=f_translate)
            print(final_epoch_output_info, file=f_translate)

            # # 画图
            # plt.figure(figsize=(6, 6))
            # plt.hist(full_coeff[_i].reshape(-1)[terms_to_keep], bins=50)
            # plt.savefig(f'tmp_distribution_{target_insulin_time_list[_i]}.png')
            # # plt.savefig(f'tmp_distribution_{target_insulin_time_list[_i]}.pdf')

            _sort = np.argsort(each.reshape(-1))
            _predict_baseline_diff = model_prediction[_i] - model_baseline_prediction[_i]

            _min_number = 5
            _max_number = -5

            reallocate_number = True
            while reallocate_number:
                _sort_min = _sort[0:_min_number]
                _sort_max = _sort[_max_number:]
                # 影响与差异同向（同符号）
                _predict_baseline_diff = model_prediction[_i] - model_baseline_prediction[_i]
                print(_min_number, _max_number, _predict_baseline_diff)

                _all_effect = np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max])
                if _predict_baseline_diff * (_all_effect) > 0:
                    # and abs(_predict_baseline_diff) > abs(_all_effect):
                    reallocate_number = False
                elif (_predict_baseline_diff > 0 and _all_effect < 0):
                    # or (_predict_baseline_diff < 0 and _all_effect < 0 and abs(_predict_baseline_diff) < abs(_all_effect)):
                    reallocate_number = True
                    _max_number = _max_number - 1  # 给all effect增加正数
                    _min_number = max(_min_number - 1, 0)
                    if _max_number == -10:  # 超过边界
                        reallocate_number = False
                        _sort_min = _sort[0:_min_number]
                        _sort_max = _sort[_max_number:]
                        _predict_baseline_diff = model_prediction[_i] - model_baseline_prediction[_i]
                        _all_effect = np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max])
                    else:
                        continue
                elif (_predict_baseline_diff < 0 and _all_effect > 0):
                    # or (_predict_baseline_diff > 0 and _all_effect > 0 and abs(_predict_baseline_diff) < abs(_all_effect)):
                    reallocate_number = True
                    _max_number = min(_max_number + 1, -1)  # 给 all effect 负向
                    _min_number = _min_number + 1
                    if _min_number == 9:  # 超过边界
                        reallocate_number = False
                        _sort_min = _sort[0:_min_number]
                        _sort_max = _sort[_max_number:]
                        _predict_baseline_diff = model_prediction[_i] - model_baseline_prediction[_i]
                        _all_effect = np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max])
                    else:
                        continue

                print('pass flag 1', reallocate_number, _predict_baseline_diff, _all_effect)
                print(_min_number, _max_number)

                print('Unnormalized score sum',
                      np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max]),
                      file=f_translate)

                _normalize_factor = _predict_baseline_diff / _all_effect

                # print(_normalize_factor, _predict_baseline_diff, _all_effect)

                # print(_sort_min, _sort_max)

                for _min_index in _sort_min:
                    # print('_min_index', _min_index)
                    print(_text_prefix_index, end=' ', file=f_translate)
                    _text_prefix_index += 1
                    _row_num = int(_min_index / each.shape[0])
                    _name_row = super_index_to_name[_row_num]
                    _col_num = int(_min_index % each.shape[0])

                    _narrow_row_num = normal_index_to_narrow_index[_row_num]
                    _narrow_col_num = normal_index_to_narrow_index[_col_num]
                    print(_narrow_row_num, _narrow_col_num)

                    _name_col = super_index_to_name[_col_num]
                    _name_row_value = sample_insulin_data_flatten[0,
                                      super_index_to_index[int(_row_num)][0]:
                                      super_index_to_index[int(_row_num)][1]]
                    _name_col_value = sample_insulin_data_flatten[0,
                                      super_index_to_index[int(_col_num)][0]:
                                      super_index_to_index[int(_col_num)][1]]

                    if _narrow_row_num == _narrow_col_num:
                        model.translate(_narrow_row_num, _narrow_col_num, each, _min_index, _predict_baseline_diff,
                                        f_out=f_translate)
                        # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _min_index, f_translate, _predict_baseline_diff)
                    else:
                        model.translate(_narrow_row_num, _narrow_col_num, each, _min_index, _predict_baseline_diff,
                                        f_out=f_translate)
                        # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _min_index, f_translate, _predict_baseline_diff)
                for _max_index in _sort_max:
                    # print('_max_index', _max_index)
                    print(_text_prefix_index, end=' ', file=f_translate)
                    _text_prefix_index += 1
                    _row_num = int(_max_index / each.shape[0])
                    _name_row = super_index_to_name[_row_num]
                    _col_num = int(_max_index % each.shape[0])
                    _name_col = super_index_to_name[_col_num]
                    _name_row_value = sample_insulin_data_flatten[0,
                                      super_index_to_index[int(_row_num)][0]:
                                      super_index_to_index[int(_row_num)][1]]
                    _name_col_value = sample_insulin_data_flatten[0,
                                      super_index_to_index[int(_col_num)][0]:
                                      super_index_to_index[int(_col_num)][1]]
                    _narrow_row_num = normal_index_to_narrow_index[_row_num]
                    _narrow_col_num = normal_index_to_narrow_index[_col_num]
                    print(_narrow_row_num, _narrow_col_num)

                    if _narrow_row_num == _narrow_col_num:
                        model.translate(_narrow_row_num, _narrow_col_num, each, _max_index, _predict_baseline_diff,
                                        f_out=f_translate)
                        # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _max_index, f_translate, _predict_baseline_diff)
                    else:
                        model.translate(_narrow_row_num, _narrow_col_num, each, _max_index, _predict_baseline_diff,
                                        f_out=f_translate)
                        # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _max_index, f_translate, _predict_baseline_diff)

                # # 画图
                # plt.figure(figsize=(10, 10))
                # plt.imshow(full_coeff_narrow[_i], cmap='seismic')
                # plt.colorbar()
                # plt.savefig(f'tmp_{target_insulin_time_list[_i]}.png')
                # # plt.savefig(f'tmp_{target_insulin_time_list[_i]}.pdf')


def translate_fake(overlap_translate_normal_index, model, full_coeff, full_coeff_fake, f_translate_fake,
                   model_baseline_prediction, model_prediction, _predict_residual,
                   target_insulin_time_list, y_insulin, final_epoch_output_info, super_index_to_name,
                   normal_index_to_narrow_index, sample_insulin_data_flatten, super_index_to_index):

    _overlap_translate_normal_index_matrix = np.zeros((full_coeff.shape[1], full_coeff.shape[2]))
    for _i in range(_overlap_translate_normal_index_matrix.shape[0]):
        for _j in range(_overlap_translate_normal_index_matrix.shape[1]):
            if _i in overlap_translate_normal_index and _j in overlap_translate_normal_index:
                _overlap_translate_normal_index_matrix[_i, _j] = 1
    _overlap_translate_normal_index_matrix = _overlap_translate_normal_index_matrix.reshape(-1)
    overlap_translate_normal_index = list(np.where(_overlap_translate_normal_index_matrix==1)[0])

    #
    _text_prefix_index = 0
    for _i, each_fake in enumerate(full_coeff_fake):
        print('Predict time', target_insulin_time_list[_i],
              '基线剂量%.2f' % model_baseline_prediction[_i],
              '模型预测剂量%.2f' % model_prediction[_i],
              '实际剂量%.2f' % y_insulin[0, target_insulin_time_list[_i]].numpy(),
              'fit residual%.2f' % _predict_residual[_i],
              file=f_translate_fake)
        print(final_epoch_output_info, file=f_translate_fake)

        # print("each_fake.shape", each_fake.shape)  # each_fake.shape (750, 750)
        _sort = np.argsort(each_fake.reshape(-1))
        print(each_fake.reshape(-1).shape)
        _predict_baseline_diff = model_prediction[_i] - model_baseline_prediction[_i]

        _min_number = 5
        _max_number = -5

        reallocate_number = True
        reallocate_number_maxtry = 10

        print("init allocate", _sort[0:_min_number], _sort[_max_number:])

        while reallocate_number and reallocate_number_maxtry > 0:
            _sort_min = _sort[0:_min_number]
            _sort_max = _sort[_max_number:]

            _sort_min = set(_sort_min).intersection(set(overlap_translate_normal_index))
            _sort_max = set(_sort_max).intersection(set(overlap_translate_normal_index))

            if len(_sort_min)==5 and len(_sort_max)==5:
                reallocate_number=False
            else:
                if len(_sort_min)<5:
                    _min_number += 1
                if len(_sort_max)<5:
                    _max_number -= 1
            reallocate_number_maxtry -= 1

        _sort_min = np.array(list(_sort_min))
        _sort_max = np.array(list(_sort_max))

        if len(set(_sort_min).intersection(set(_sort_max)))>0:
            # there are intersection between sortmin and sortmax number
            return "FailToGenerateFakeTranslation"
        else:
            each = full_coeff[_i]

            if len(_sort_min)==0 or len(_sort_max) == 0:
                return "FailEmptyOverlap"
            _all_effect = np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max])

            print('pass flag 1', reallocate_number, _predict_baseline_diff, _all_effect)
            print("_min_number, _max_number", _min_number, _max_number)

            print('Unnormalized score sum',
                  np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max]),
                  file=f_translate_fake)

            _normalize_factor = _predict_baseline_diff / _all_effect

            # print(_normalize_factor, _predict_baseline_diff, _all_effect)

            # print(_sort_min, _sort_max)

            for _min_index in _sort_min:
                # print('_min_index', _min_index)
                print(_text_prefix_index, end=' ', file=f_translate_fake)
                _text_prefix_index += 1
                _row_num = int(_min_index / each.shape[0])
                _name_row = super_index_to_name[_row_num]
                _col_num = int(_min_index % each.shape[0])

                _narrow_row_num = normal_index_to_narrow_index[_row_num]
                _narrow_col_num = normal_index_to_narrow_index[_col_num]
                print(_narrow_row_num, _narrow_col_num)

                _name_col = super_index_to_name[_col_num]
                _name_row_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_row_num)][0]:
                                  super_index_to_index[int(_row_num)][1]]
                _name_col_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_col_num)][0]:
                                  super_index_to_index[int(_col_num)][1]]

                if _narrow_row_num == _narrow_col_num:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _min_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _min_index, f_translate, _predict_baseline_diff)
                else:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _min_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _min_index, f_translate, _predict_baseline_diff)
            for _max_index in _sort_max:
                # print('_max_index', _max_index)
                print(_text_prefix_index, end=' ', file=f_translate_fake)
                _text_prefix_index += 1
                _row_num = int(_max_index / each.shape[0])
                _name_row = super_index_to_name[_row_num]
                _col_num = int(_max_index % each.shape[0])
                _name_col = super_index_to_name[_col_num]
                _name_row_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_row_num)][0]:
                                  super_index_to_index[int(_row_num)][1]]
                _name_col_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_col_num)][0]:
                                  super_index_to_index[int(_col_num)][1]]
                _narrow_row_num = normal_index_to_narrow_index[_row_num]
                _narrow_col_num = normal_index_to_narrow_index[_col_num]
                print(_narrow_row_num, _narrow_col_num)

                if _narrow_row_num == _narrow_col_num:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _max_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _max_index, f_translate, _predict_baseline_diff)
                else:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _max_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _max_index, f_translate, _predict_baseline_diff)

    return "Success"


def translate_fake_random_sentence(true_translation_sentence_list_full, random_translation_sentence_list_full, model,
                                   full_coeff, full_coeff_fake, f_translate_fake,
                                   model_baseline_prediction, model_prediction, _predict_residual,
                                   target_insulin_time_list, y_insulin, final_epoch_output_info, super_index_to_name,
                                   normal_index_to_narrow_index, sample_insulin_data_flatten, super_index_to_index):

    _overlap_translate_normal_index_matrix = np.zeros((full_coeff.shape[1], full_coeff.shape[2]))
    for _i in range(_overlap_translate_normal_index_matrix.shape[0]):
        for _j in range(_overlap_translate_normal_index_matrix.shape[1]):
            if len(true_translation_sentence_list_full[_i])>0 and len(true_translation_sentence_list_full[_j])>0:
                _overlap_translate_normal_index_matrix[_i, _j] = 1
    _overlap_translate_normal_index_matrix = _overlap_translate_normal_index_matrix.reshape(-1)
    overlap_translate_normal_index = list(np.where(_overlap_translate_normal_index_matrix==1)[0])

    print("overlap_translate_normal_index", overlap_translate_normal_index, max(overlap_translate_normal_index),
          min(overlap_translate_normal_index), len(overlap_translate_normal_index))

    _text_prefix_index = 0
    for _i, each in enumerate(full_coeff):
        print('Predict time', target_insulin_time_list[_i],
              '基线剂量%.2f' % model_baseline_prediction[_i],
              '模型预测剂量%.2f' % model_prediction[_i],
              '实际剂量%.2f' % y_insulin[0, target_insulin_time_list[_i]].numpy(),
              'fit residual%.2f' % _predict_residual[_i],
              file=f_translate_fake)
        print(final_epoch_output_info, file=f_translate_fake)

        # print("each_fake.shape", each_fake.shape)  # each_fake.shape (750, 750)
        each_positive_index = np.where(each.reshape(-1)>0)[0].tolist()
        each_negative_index = np.where(each.reshape(-1)<0)[0].tolist()
        each_zero_index = np.where(each.reshape(-1)==0)[0].tolist()

        print(len(each_positive_index), len(each_negative_index), len(each_zero_index))

        _sort_positive = set(list(each_positive_index)).intersection(set(overlap_translate_normal_index))
        _sort_negative = set(list(each_negative_index)).intersection(set(overlap_translate_normal_index))
        _sort_zero = set(list(each_zero_index)).intersection(set(overlap_translate_normal_index))

        print(len(set(_sort_positive).intersection(set(_sort_zero))))

        print(len(_sort_positive), len(_sort_negative), len(_sort_zero))

        # _sort = np.argsort(each.reshape(-1))
        # _sort_value = np.sort(each.reshape(-1))

        _predict_baseline_diff = model_prediction[_i] - model_baseline_prediction[_i]

        if _predict_baseline_diff>0:
            _sort = list(_sort_positive)
        else:
            _sort = list(_sort_negative)
        assert len(_sort) > 10
        np.random.shuffle(_sort)

        _min_number = 5
        _max_number = -5

        reallocate_number = True
        reallocate_number_maxtry = 10

        print("init allocate", _sort[0:_min_number], _sort[_max_number:])

        while reallocate_number and reallocate_number_maxtry > 0:
            print("_min_number _max_number", _min_number, _max_number)
            _sort_min = _sort[0:_min_number]
            _sort_max = _sort[_max_number:]

            _sort_min = set(_sort_min).intersection(set(overlap_translate_normal_index))
            _sort_max = set(_sort_max).intersection(set(overlap_translate_normal_index))
            print("_sort_min _sort_max", _sort_min, _sort_max)

            if len(_sort_min) == 5 and len(_sort_max) == 5:
                reallocate_number = False
            else:
                if len(_sort_min) < 5:
                    _min_number += 1
                if len(_sort_max) < 5:
                    _max_number -= 1
            reallocate_number_maxtry -= 1

        _sort_min = np.array(list(_sort_min))
        _sort_max = np.array(list(_sort_max))

        if len(set(_sort_min).intersection(set(_sort_max))) > 0:
            # there are intersection between sortmin and sortmax number
            print("FailToGenerateFakeTranslation")
            assert 1==0  # 不应该有这种情况
            return "FailToGenerateFakeTranslation"
        else:
            each = full_coeff[_i]

            if len(_sort_min) == 0 or len(_sort_max) == 0:
                print("FailEmptyOverlap")
                return "FailEmptyOverlap"
            _all_effect = np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max])

            print('pass flag 1', reallocate_number, _predict_baseline_diff, _all_effect)
            print("_min_number, _max_number", _min_number, _max_number)

            print('Unnormalized score sum',
                  np.sum(each.reshape(-1)[_sort_min]) + np.sum(each.reshape(-1)[_sort_max]),
                  file=f_translate_fake)

            _normalize_factor = _predict_baseline_diff / _all_effect

            # resort
            _value_helper = []
            _sort_min_and_max = np.hstack((_sort_min, _sort_max))
            for _ii in _sort_min_and_max:
                _value_helper.append(each.reshape(-1)[_ii])
            _sort_min = _sort_min_and_max[np.argsort(_value_helper)][0:len(_sort_min)]
            _sort_max = _sort_min_and_max[np.argsort(_value_helper)][len(_sort_min):]


            for _min_index in _sort_min:
                # print('_min_index', _min_index)
                print(_text_prefix_index, end=' ', file=f_translate_fake)
                _text_prefix_index += 1
                _row_num = int(_min_index / each.shape[0])
                _name_row = super_index_to_name[_row_num]
                _col_num = int(_min_index % each.shape[0])

                _narrow_row_num = normal_index_to_narrow_index[_row_num]
                _narrow_col_num = normal_index_to_narrow_index[_col_num]
                print(_narrow_row_num, _narrow_col_num)

                _name_col = super_index_to_name[_col_num]
                _name_row_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_row_num)][0]:
                                  super_index_to_index[int(_row_num)][1]]
                _name_col_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_col_num)][0]:
                                  super_index_to_index[int(_col_num)][1]]

                if _narrow_row_num == _narrow_col_num:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _min_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _min_index, f_translate, _predict_baseline_diff)
                else:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _min_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _min_index, f_translate, _predict_baseline_diff)
            for _max_index in _sort_max:
                # print('_max_index', _max_index)
                print(_text_prefix_index, end=' ', file=f_translate_fake)
                _text_prefix_index += 1
                _row_num = int(_max_index / each.shape[0])
                _name_row = super_index_to_name[_row_num]
                _col_num = int(_max_index % each.shape[0])
                _name_col = super_index_to_name[_col_num]
                _name_row_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_row_num)][0]:
                                  super_index_to_index[int(_row_num)][1]]
                _name_col_value = sample_insulin_data_flatten[0,
                                  super_index_to_index[int(_col_num)][0]:
                                  super_index_to_index[int(_col_num)][1]]
                _narrow_row_num = normal_index_to_narrow_index[_row_num]
                _narrow_col_num = normal_index_to_narrow_index[_col_num]
                print(_narrow_row_num, _narrow_col_num)

                if _narrow_row_num == _narrow_col_num:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _max_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _max_index, f_translate, _predict_baseline_diff)
                else:
                    model.translate(_narrow_row_num, _narrow_col_num, each, _max_index, _predict_baseline_diff,
                                    f_out=f_translate_fake)
                    # model.translate(_row_num, _col_num, _name_row, _name_col, _name_row_value, _name_col_value, each * _normalize_factor, _max_index, f_translate, _predict_baseline_diff)

    return "Success"