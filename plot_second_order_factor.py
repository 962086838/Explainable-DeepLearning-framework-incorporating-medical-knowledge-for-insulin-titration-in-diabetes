import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def load_sample_data(path, index_x=None, index_y=None):
    print(path)
    _sample = np.load(path, allow_pickle=True)
    exam = _sample[0].reshape(-1)
    insulin = _sample[1].reshape(3, 7, -1)
    temp_insulin = _sample[2].reshape(3, 7, -1)
    sugar = _sample[3].reshape(3, 7)
    drug = _sample[4].reshape(3, 7, -1)
    days = _sample[5]

    if index_x is None and index_y is None:
        return exam, insulin, temp_insulin, sugar, drug, days
    else:
        # return the corresponding value
        return_list = []
        for each_index in [index_x, index_y]:
            if each_index < 78:
                return_list.append(exam[each_index])
            elif each_index < 78 + 21:
                return_list.append(insulin[each_index - 78])
            elif each_index < 78 + 21 + 21:
                return_list.append(temp_insulin[each_index - 78 - 21])
            elif each_index < 78 + 21 + 21 + 21:
                return_list.append(sugar.reshape(-1)[each_index - 78 - 21 - 21])
            elif each_index < 78 + 21 + 21 + 21 + 21:
                return_list.append(drug[each_index - 78 - 21 - 21 - 21])
        return return_list

def main(folder, target_time=None, index_x=None, index_y=None):
    filepaths = glob(f'{folder}/full_coeff_*.npy')
    # load coeff matrix

    # 构造需要显示的值
    X = np.arange(0, 25, step=0.5)  # X轴的坐标
    Y = np.arange(0, 25, step=0.5)  # Y轴的坐标
    # 设置每一个（X，Y）坐标所对应的Z轴的值，在这边Z（X，Y）=X+Y
    Z = np.zeros(shape=(50, 50))
    Z_count = np.zeros(shape=(50, 50))



    for each_path in filepaths[0:50]:
        file_rand_index = each_path.split('full_coeff_')[1].split('.npy')[0]
        full_coeff = np.load(each_path)
        sample_data = load_sample_data(f'{folder}/sample_input_tuple{file_rand_index}.npy', index_x=index_x, index_y=index_y)
        print(sample_data[0])  # x
        print(sample_data[1])  # y
        print(full_coeff.shape)



        if target_time is not None: # only one target time
            target_coeff = full_coeff[target_time, index_x, index_y]
            _i, _j = int(sample_data[0] / 0.5), int(sample_data[1] / 0.5)
            Z[int(sample_data[0] / 0.5), int(sample_data[1] / 0.5)] = (Z[_i, _j] * Z_count[_i, _j] + target_coeff) / (Z_count[_i, _j] + 1)
            Z_count[int(sample_data[0] / 0.5), int(sample_data[1] / 0.5)] += 1
        else:
            assert 1==0
        print(file_rand_index, each_path)

    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    bottom = np.zeros_like(X)  # 设置柱状图的底端位值
    Z = Z.ravel()  # 扁平化矩阵

    width = height = 1  # 每一个柱子的长和宽

    # 绘图设置
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    ax.bar3d(X, Y, bottom, width, height, Z, shade=True)  #
    # 坐标轴设置
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z(value)')
    plt.show()

if __name__ == "__main__":
    folder = 'assess_save_20220726'
    main(folder, target_time=0, index_x=129, index_y=129)