import warnings
import argparse
import os
import argparse
import torch
import numpy as np
from torch import Generator, randperm
import random
from torch.utils.data import DataLoader, Subset
from train import train_cnn
from model import SRUNet, CubicWeightedPSNRLoss
from data import KidneySRUData

warnings.filterwarnings('ignore')


def train_distributed(args):
    model = SRUNet(up_scale=6).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CubicWeightedPSNRLoss().cuda()
    dataset_train = KidneySRUData(data_root='/data/cxlu/srunet_liver_training_large/srunet_training')
    data_len = len(dataset_train)
    valid_len = int(data_len * 0.1)
    data_len -= valid_len
    dataset_train, dataset_val = subset_split(dataset_train, lengths=[data_len, valid_len],
                                              generator=Generator().manual_seed(args.seed))
    train_loader = DataLoader(dataset=dataset_train, num_workers=args.num_worker, batch_size=args.b, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, num_workers=args.num_worker, batch_size=args.b, pin_memory=True,
                            )
    train_cnn(train_generator=train_loader, valid_generator=val_loader, args=args, optimizer=optimizer, model=model,
              criterion=criterion)

def subset_split(dataset, lengths, generator):
    """
     """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length: offset]))
    return Subsets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--b', default=16, type=int, help='batch size')
    parser.add_argument('--epoch', default=100, type=int, help='epochs to train')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--output', default='./model_genesis_pretrain', type=str, help='output path')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='gpu indexs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_worker', type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    train_distributed(args)
