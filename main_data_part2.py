from __future__ import division
import numpy as np
import argparse
import torch
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--o', default='data/dataset_beamforming.pt', help='output file')
parser.add_argument('--num_tasks', default=3, type=int, help='number of tasks')
parser.add_argument('--num_train', default='', type=str)
parser.add_argument('--split', default=4, type=int)
args = parser.parse_args()

for s in range(args.split):
    args.o ='data/dataset_beamforming_part%d.pt' % s
    tasks_tr = []
    tasks_te = []
    tasks_inx = []
    for t in range(args.num_tasks):
        # Reading input and output sets generated from MATLAB
        filepath = 'DLCB_Dataset/DLCB_%d.mat' % (t+1)
        f = sio.loadmat(filepath)
        in_set = f['DL_in']
        out_set = f['DL_out'][:, :, s]
    #    print(in_set.shape, out_set.shape)
            
        # train test split
        num_samples = in_set.shape[0]
        num_in = in_set.shape[1]
        num_out = out_set.shape[1]
        num_train = 15000
        args.num_train = args.num_train + str(num_train) + '-'
        num_test = 1000
        np.random.seed(0)
        train_index = np.random.choice(
            range(0, num_samples), size=num_train, replace=False)
        rem_index = set(range(0, num_samples))-set(train_index)
        test_index = list(set(np.random.choice(
            list(rem_index), size=num_test, replace=False)))
        In_train = in_set[train_index, :]
        In_test = in_set[test_index, :]
        Out_train = out_set[train_index, :]
        Out_test = out_set[test_index, :]

        Xtrain = torch.from_numpy(In_train).float()
        Ytrain = torch.from_numpy(Out_train).float()
        tasks_tr.append([t, Xtrain.clone(), Ytrain.clone()])
        print(Xtrain.shape, Ytrain.shape)

        Xtest = torch.from_numpy(In_test).float()
        Ytest = torch.from_numpy(Out_test).float()
        tasks_te.append([t, Xtest.clone(), Ytest.clone()])
        print(Xtest.shape, Ytest.shape)
        
        tasks_inx.append(test_index)
        
    torch.save([tasks_tr, tasks_te, args, tasks_inx], args.o)
