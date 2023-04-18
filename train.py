# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: peterli <j2457li@uwaterloo.ca>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/17 21:43:13 by j2457li           #+#    #+#              #
#    Updated: 2023/04/17 22:05:41 by peterli          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys, os
import argparse
from training.training_api import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the Classification Model on Cifar100 Dataset",
        epilog="Author: Peter Li")
    parser.add_argument("-n", "--net", type=str, required=True, help="Neural Net Name")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument("-w", "--warm_up", type=int, default=0, help="Number of Epochs for Warm-Up Training")
    parser.add_argument("-l", "--lr", type=float, default=0.01, help="Initial Learning Rate")
    parser.add_argument("--gpu", default=False, action="store_true", help="Enable CUDA for GPU Training")

    args = parser.parse_args()
    net_name = args.net
    gpu = args.gpu
    batch_size = args.batch_size
    warm_epoch = args.warm_up
    lr = args.lr
    train(net_name, batch_size, warm_epoch, lr, gpu)
