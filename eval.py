import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from utils.load_model import load_model
from data import data_loaders

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')

    args = parser.parse_args()
    net_name = args.net
    batch_size = args.batch_size
    weights = args.weights

    net = load_model(net_name)

    cifar100_test_loader = data_loaders.testing_data_loader(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD,
                                                            batch_size=batch_size, num_workers=4, shuffle=True)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in tqdm(enumerate(cifar100_test_loader)):
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
