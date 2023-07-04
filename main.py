import torch
import os
from dataset.task_cv import TaskA
from dataset.createDataset import load_data_XLDA
from train.trainer import train_Model
from models.X_LDA import X_LDA


class argparse():
    pass


def cross_val_XLDA(module, Sd, Lm, Md, task_pair, args):
    k = 5
    epoch1 = args.EPOCHs  # 80
    _LR = args.LRs
    batchsize = args.BatchSizes

    for i in range(k):
        Sm = task_pair.sim_seq_cos
        train_loader = load_data_XLDA(task_pair.train_pairs[i], task_pair.get_k_A(i), Sm, Sd, batchsize, Lm,
                                      Md)
        model = module(args.conv_num).cuda()
        model.train()
        train_Model(model, epoch1, train_loader, _LR, args)


def main():
    args = argparse()
    args.device = torch.device("cuda:0")

    task_pair = TaskA()
    A_ori, Sd, Lm, Md, Sl_seq = task_pair.get_association_similarity_martix()

    # our model
    print("X-LDA")

    args.EPOCHs = 90
    args.LRs = 0.0005
    args.BatchSizes = 128
    args.conv_num = 64

    cross_val_XLDA(X_LDA, Sd, Lm, Md, task_pair, args)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()
