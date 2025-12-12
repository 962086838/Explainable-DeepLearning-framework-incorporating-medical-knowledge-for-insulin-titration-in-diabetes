import torch
import time
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam, Adagrad, RMSprop
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import copy

lhs = torch.tensor([[1., 1, 1],
                   [2, 3, 4],
                   [3, 5, 2],
                   [4, 2, 5],
                   [5, 4, 3]])
rhs = torch.tensor([[-10., -3],
                    [12, 14],
                    [14, 12],
                    [16, 16],
                    [18, 16]])
w_cuda = torch.ones((5, 2))

# SVRG++ epoch early stop
mse_f = torch.nn.MSELoss()
n = w_cuda.shape[0]
m0 = int(n / 4)
eta = 0.01
S = 10
ep_loss_list = []
# x_s = torch.rand_like(w_cuda, requires_grad=True)
x_s = 0.01*torch.rand_like(w_cuda, requires_grad=True) + w_cuda
x_s_t = Variable(x_s.data, requires_grad=True)
start_time = time.time()
m_s_list = []
for s in range(1, S + 1):
    x_s = Variable(x_s.data, requires_grad=True)
    loss = mse_f(torch.matmul(lhs, x_s), rhs)
    ep_loss_list.append(loss.data.item())
    full_gradient = torch.autograd.grad(loss, x_s, retain_graph=False)[0]
    m_s = 2 ** s * m0
    m_s_list.append(m_s)
    k = 0
    T = np.sum(m_s_list)

    random_permuation = torch.randperm(n)
    x_s_t_list = []
    if s == 1:  # first epoch, create
        this_epoch_diff_list = []
        last_epoch_diff_list = []
    else:  # not first epoch, copy
        last_epoch_diff_list = [each for each in this_epoch_diff_list]
        this_epoch_diff_list = []

    early_stop = False
    for t in range(m_s):
        data_index = random_permuation[t % w_cuda.shape[0]]
        grad_1 = torch.autograd.grad(mse_f(torch.matmul(lhs[data_index], x_s_t), rhs[data_index]), x_s_t, retain_graph=False)[0]
        grad_2 = torch.autograd.grad(mse_f(torch.matmul(lhs[data_index], x_s), rhs[data_index]), x_s, retain_graph=False)[0]
        diff_t = grad_1 - grad_2
        kesi = diff_t + full_gradient
        k = k + 1
        eta_s_tp1 = eta * np.sqrt(T) / np.sqrt(2 * T - k)
        x_s_t = x_s_t - kesi * eta_s_tp1
        x_s_t_list.append(x_s_t.detach().cpu().numpy())

        this_epoch_diff_list.append(torch.norm(diff_t, p=2))
        if len(this_epoch_diff_list) > int(n/4):
            this_epoch_diff_list.pop(0)
            assert len(this_epoch_diff_list)==int(n/4)
        average_diff = torch.mean(this_epoch_diff_list)
        if torch.sum(last_epoch_diff_list<average_diff) > int(n/2):
            early_stop = True
        if early_stop:
            break

    assert len(x_s_t_list) == m_s
    _sum = torch.zeros(x_s_t_list[0].shape)
    for each in x_s_t_list:
        _sum += each
    x_s = _sum / len(x_s_t_list)
    x_s = Variable(x_s.data, requires_grad=True).cuda()

    x_s_t = Variable(x_s_t.data, requires_grad=True)
    print(mse_f(torch.matmul(lhs, x_s), rhs).data.item(), x_s[0:5, 0])
print('time', time.time() - start_time)
print(mse_f(torch.matmul(lhs, x_s), rhs))
print(x_s[0:5, 0])
plt.plot(ep_loss_list)