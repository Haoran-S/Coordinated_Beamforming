# ###############################################
# This file was written for ``Learning to Continuously Optimize" [1].
# DNN model part is modified from ``Learning to Optimize" [2].
# Codes have been tested successfully on Python 3.6.0.
#
# References:
# [1] Haoran Sun, Wenqiang Pu, Minghe Zhu, Xiao Fu, Tsung-Hui Chang,
# Mingyi Hong, "Learning to Continuously Optimize Wireless Resource In
# Episodically Dynamic Environment",
# arXiv preprint arXiv:2011.07782 (2020).
#
# [2] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu
# and Nikos D. Sidiropoulos, “Learning to Optimize: Training Deep
# Neural Networks for Wireless Resource Management”,
# IEEE Transactions on Signal Processing 66.20 (2018): 5438-5453.
#
# version 1.0 -- Oct. 2020.
# Haoran Sun (sunhr1993 @ gmail.com)
# All rights reserved.
# ###############################################

import importlib
import argparse
import random
import time
import os
import numpy as np
import torch
from scipy.io import savemat


class Continuum:
    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)

        self.permutation = []
        for t in range(n_tasks):
            N = data[t][1].size(0)
            for _ in range(args.n_epochs):
                task_p = [[t, i] for i in range(N)]
                random.shuffle(task_p)
                self.permutation += task_p
#            print("Task", t, "Samples are", N)

        self.length = len(self.permutation)
        self.current = 0
#        print("total length", self.length)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]


def eval_tasks(model, tasks, args, ep, d_idx):
    """
    evaluates the model on all tasks
    """
    model.eval()
    MSEloss = torch.nn.MSELoss()
    mseloss = []

    for i, task in enumerate(tasks):
        t = i
        xb = task[1]
        yb = task[2]

        if args.cuda:
            xb = xb.cuda()
        output = model(xb, t).data.cpu()
       
        mseloss.append(MSEloss(output, yb.cpu()).item())
        mdic = {"predict": output.numpy(), "label": yb.numpy(), "user_index": d_idx[i]}
        savemat("result/pred_%s_t%d_ep%d_part%d.mat" % (args.model+args.mode, ep, i, args.part), mdic)
    
    print('mse:', mseloss)
        
    return mseloss


def life_experience(model_o, continuum, x_te, args, d_idx, accumulate_train=False):
    result_t_mse = []
    result_t_rate = []
    result_t_ratio = []
    time_all = []
    result_all = []  # avg performance on all test samples
    current_task = 0
    time_start = time.time()
    time_spent = 0
    model = model_o
    loss2save = []
    time2save = []

    for (i, (v_x, t, v_y)) in enumerate(continuum):
        if accumulate_train:
#            model = model_o
            if i == 0:
                v_x_acc = v_x
                v_y_acc = v_y
            else:
                v_x_acc = torch.cat((v_x_acc, v_x), 0)
                v_y_acc = torch.cat((v_y_acc, v_y), 0)
            v_x = v_x_acc
            v_y = v_y_acc
            
            perm_index = torch.randperm(v_x.size()[0])
            v_x = v_x[perm_index]
            v_y = v_y[perm_index]

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        time_start = time.time()

        model.train()
        model.observe(v_x, t, v_y, loss_type='MSE', x_te=x_te, x_tr=x_tr)

        time_end = time.time()
        time_spent = time_spent + time_end - time_start

        mseloss = eval_tasks(model, x_te, args, i, d_idx)
        loss2save.append(mseloss)
        time2save.append(time_spent)

    return loss2save, time2save


def load_datasets(args):
    d_tr, d_te, args, d_idx = torch.load(args.data_file + '_part' + str(args.part) + '.pt')
    n_inputs = d_tr[0][1].size(1)
    n_outputs = d_tr[0][2].size(1)
#    print(args)
    return d_tr, d_te, n_inputs, n_outputs, len(d_tr), d_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--hidden_layers', type=str, default='200-80-80',
                        help='hidden neurons at each layer')
    parser.add_argument('--unsupervised', type=str, default='n',
                        help='use unsupervised learning')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Number of iterations per batch')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='batch size')
    parser.add_argument('--mini_batch_size', type=int, default=100,
                        help='mini batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=1,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='./',
                        help='path where data is located')
    parser.add_argument('--data_file', default='data/dataset_balance.pt',
                        help='data file')
    parser.add_argument('--file_ext', default='',
                        help='file name extention')

    # general experiments parameters
    parser.add_argument('--n_memories', type=int, default=2000,
                        help='memory size')
    parser.add_argument('--age', type=float, default=0,
                        help='consider age for sample selection')
    parser.add_argument('--mode', type=str, default='online',
                        help='feed data online or joint training')
    parser.add_argument('--noise', type=float, default=1.0,
                        help='noise level for sum-rate calculation')

    # min-max parameter
    parser.add_argument('--weight_ini', type=str, default='pra',
                        help='pra, rand, mean')
    parser.add_argument('--eval_metric', type=str, default='ratio',
                        help='ratio or mse')
    parser.add_argument('--dual_stepsize', default=0.00000001, type=float,
                        help='dual stepsize for PGD in min max CL')

    # GSS parameter
    parser.add_argument('--subselect', type=int, default=1,
                        help='first subsample from recent memories')
    parser.add_argument('--repass', type=int, default=0,
                        help='make a repass over the previous da<ta')
    parser.add_argument('--eval_memory', type=str, default='no',
                        help='compute accuracy on memory')
    parser.add_argument('--n_sampled_memories', type=int, default=0,
                        help='number of sampled_memories per task')
    parser.add_argument('--n_constraints', type=int, default=0,
                        help='number of constraints to use during online training')
    parser.add_argument('--change_th', type=float, default=0.0,
                        help='gradients similarity change threshold for re-estimating the constraints')
    parser.add_argument('--slack', type=float, default=0,
                        help='slack for small gradient norm')
    parser.add_argument('--normalize', type=str, default='no',
                        help='normalize gradients before selection')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
                        
                        
    parser.add_argument('--part', default=0, type=int)

    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    if args.mini_batch_size == 0:
        args.mini_batch_size = args.batch_size  # no mini iterations

    # initialize seeds
    print('------- Start -------\n')
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks, d_idx = load_datasets(args)

    # set up continuum
    continuum = Continuum(x_tr, args)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)

    # # set up file name for inbetween saving
    model.fname = args.model + '_' + args.mode + args.file_ext
    model.fname = os.path.join(args.save_path, model.fname)

    if args.cuda:
        model.cuda()

    if args.mode == 'online':
        # run model on continuum
        loss2save, time2save = life_experience(model, continuum, x_te, args, d_idx, accumulate_train=False)
    elif args.mode == 'joint':
        # run model on entire dataset
        loss2save, time2save = life_experience(model, continuum, x_te, args, d_idx, accumulate_train=True)
    else:
        raise AssertionError(
            "args.mode should be one of 'online', 'joint'.")

    # save mseloss and cpu time
    mdic = {"loss": loss2save, "time": time2save}
    savemat("result/save_%s_part%d.mat" % (args.model+args.mode, args.part), mdic)
            
    
