import os.path
import sys
import time
from datetime import datetime
from data import data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from utils.learning_rate import WarmUpLR
from tensorboardX import SummaryWriter
from utils.load_model import load_model
from models import *

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def training_api(net, training_data_loader, optimizer, loss_function,
                 warmup_scheduler, batch_size, warm_up_epoch_num, epoch_num, writer, use_gpu):

    start = time.time()
    # Puts the Neural Net in Training Mode
    net.train()

    # Iterates through each batch
    for batch_index, (images, labels) in enumerate(training_data_loader):
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        # Clears gradient on each parameter/layer/weight
        optimizer.zero_grad()
        # Calls forward function, thus making a prediction
        outputs = net(images)
        # Compute loss
        loss = loss_function(outputs, labels)
        # Calculate for gradient
        loss.backward()
        # Update the gradient
        optimizer.step()

        n_iter = (epoch_num - 1) * len(training_data_loader) + batch_index + 1
        # Starting from here is what I don't understand why doing
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch_num,
            trained_samples=batch_index * batch_size + len(images),
            total_samples=len(training_data_loader.dataset)
        ))

        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch_num <= warm_up_epoch_num:
            warmup_scheduler.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch_num, finish - start))


@torch.no_grad()
def eval_training(net, testing_data_loader, loss_function, writer, epoch=0, tb=True, use_gpu=False):
    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in testing_data_loader:
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(testing_data_loader.dataset),
        correct.float() / len(testing_data_loader.dataset),
        finish - start
    ))
    print()

    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(testing_data_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(testing_data_loader.dataset), epoch)

    return correct.float() / len(testing_data_loader.dataset)


def train(net_name="resnet18", batch_size=128, warm_epoch=1, lr=0.1, use_gpu=False):

    # Initialize Neural Net, Loss Function, Optimizer, and LR Scheduler
    net = load_model(net_name)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # Initialize Data Loaders
    training_data_loader = data_loaders.training_data_loader(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD,
                                                             batch_size=batch_size, num_workers=4, shuffle=True)
    testing_data_loader = data_loaders.testing_data_loader(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD,
                                                           batch_size=batch_size, num_workers=4, shuffle=True)
    # Initialize WarmUp LR Scheduler
    iter_per_epoch = len(training_data_loader)
    num_of_warm_up_iters = iter_per_epoch * warm_epoch
    warmup_scheduler = WarmUpLR(optimizer, num_of_warm_up_iters)

    # Check Output Path
    checkpoint_path = "./checkpoints"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # Initialize Summary Writer
    writer = SummaryWriter(log_dir=os.path.join("./log", net_name, datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')))

    # Start Training Epoch by Epoch
    best_acc = 0.0
    for epoch in range(1, 30 + 1):
        if epoch > warm_epoch:
            train_scheduler.step(epoch)

        # Does the Training for One Epoch
        training_api(net, training_data_loader, optimizer, loss_function,
                     warmup_scheduler, batch_size, warm_epoch, epoch, writer, use_gpu)

        # Test this Epoch
        acc = eval_training(net, testing_data_loader, loss_function, writer, epoch, tb=True, use_gpu=False)

        # Start Saving Best Performance Models after LR Decays to 0.01
        if epoch > 120 and best_acc < acc:
            weights_path = checkpoint_path.format(net=net_name, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % 1:
            weights_path = checkpoint_path.format(net=net_name, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
