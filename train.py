import sys, os
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from training.training_api import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a ResNet Model (with variable number of layers) on Cifar100 Dataset")
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--warm_up', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    net_name = args.net
    batch_size = args.batch_size
    warm_epoch = args.warm_up
    lr = args.lr
    train(net_name, batch_size, warm_epoch, lr)
