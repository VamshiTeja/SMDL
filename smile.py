import argparse, time, os
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import *
from datasets import cifar
from helper.utils import *


def learn_incrementally(args, gpus, datatset='CIFAR'):
    if datatset == 'CIFAR':
        num_classes = 100
        # Create the class-id list
        classes = np.arange(num_classes)

    print 'Training starting.'
    train_start_time = time.time()

    # Each round is one iteration of the whole experiment. This is done to measure the robustness of the network.
    for round_count in range(args.repeat_rounds):
        # Each episode contains a set of classes that are trained at once.
        # Different episodes contains different classes and classes in all episodes is equal to the total classes.
        class_per_episode = args.class_per_episode
        num_episodes = num_classes / class_per_episode

        # Shuffle the class order for each episode.
        np.random.shuffle(classes)

        # Initialize the model
        model = resnet32()
        model.apply(weights_init)
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        for episode_count in np.arange(num_episodes):
            ep_class_set = classes[episode_count*class_per_episode: (episode_count+1)*class_per_episode]  # New class in this episode
            cumm_class_set = classes[0: (episode_count+1)*class_per_episode]  # All classes upto and including this episode

            norm = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(), norm
                                                   ])
            test_transforms = transforms.Compose([transforms.ToTensor(), norm])

            train_dataset = cifar.CIFAR100(root='./datasets/', train=True, download=False, transform=train_transforms,
                                           class_list=cumm_class_set)
            test_dataset = cifar.CIFAR100(root='./datasets/', train=False, transform=test_transforms,
                                          class_list=cumm_class_set)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=2)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False,
                                                      num_workers=2)

            train_accs = []
            test_accs = []
            # Repeat for each epoch
            for epoch_count in range(args.epochs):
                adjust_lr(epoch_count, optimizer, args.learning_rate)

                start_time = time.time()

                train_acc = train(train_loader, model, criterion, optimizer, epoch_count, args.epochs, episode_count, num_episodes,
                      round_count, args.repeat_rounds)
                test_acc = test(test_loader, model, epoch_count, args.epochs, episode_count, num_episodes,
                      round_count, args.repeat_rounds)

                train_accs.append(train_acc)
                test_accs.append(test_acc)
                print 'Time per epoch: {0:.4f}s \n'.format(time.time() - start_time)
            plot_per_epoch_accuracies(train_accs, test_accs, episode_count, round_count)

    print 'Training complete. Total time: {0:.4f} mins.'.format((time.time() - train_start_time)/60)


def train(train_loader, model, criterion, optimizer, epoch_count, max_epoch, episode_count, max_episodes,
          round_count, max_rounds, logging_freq=10, detailed_logging=False):
    losses = Metrics()
    top1 = Metrics()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()

        output = model(Variable(input))
        loss = criterion(output, Variable(target))

        acc = compute_accuracy(output.data, target)[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(acc.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % logging_freq == 0 and detailed_logging:
            print 'Round: {0:3d}/{1}\t Episode: {2:3d}/{3} \t Epoch {4:3d}/{5}[{6:3d}/{7}] ' \
                  '\t Loss: {loss.val:.4f}({loss.avg:.4f}) ' \
                  '\t Training_Accuracy: {accuracy.val:.4f}({accuracy.avg:.4f})'.format(round_count+1, max_rounds, episode_count+1, max_episodes,
                                                                             epoch_count+1, max_epoch, i, len(train_loader),
                                                                             loss=losses, accuracy=top1)

    print 'Round: {0:3d}/{1}\t Episode: {2:3d}/{3} \t Epoch {4:3d}/{5}' \
          '\t Loss: {loss.val:.4f}({loss.avg:.4f}) ' \
          '\t Training_Accuracy: {accuracy.val:.4f}({accuracy.avg:.4f})'.format(round_count + 1, max_rounds,
                                                                                episode_count + 1, max_episodes,
                                                                                epoch_count + 1, max_epoch,
                                                                                loss=losses, accuracy=top1)
    return top1.avg


def test(test_loader, model, epoch_count, max_epoch, episode_count, max_episodes, round_count, max_rounds, logging_freq=10, detailed_logging=False):
    top1 = Metrics()
    model.eval()

    for i, (input, target) in enumerate(test_loader):
        input, target = input.cuda(), target.cuda()
        output = model(Variable(input))
        acc = compute_accuracy(output.data, target)[0]
        top1.update(acc.item(), input.size(0))

        if i % logging_freq == 0 and detailed_logging:
            print 'Round: {0:3d}/{1}\t Episode: {2:3d}/{3} \t Epoch {4:3d}/{5}[{6:3d}/{7}] ' \
                  '\t Test_Accuracy: {accuracy.val:.4f}({accuracy.avg:.4f})'.format(round_count+1, max_rounds, episode_count+1, max_episodes,
                                                                             epoch_count+1, max_epoch, i, len(test_loader),
                                                                             accuracy=top1)

    print 'Round: {0:3d}/{1}\t Episode: {2:3d}/{3} \t Epoch {4:3d}/{5}' \
          '\t Test_Accuracy: {accuracy.avg:.4f}'.format(round_count + 1, max_rounds,
                                                                            episode_count + 1, max_episodes,
                                                                            epoch_count + 1, max_epoch,
                                                                            accuracy=top1)
    return top1.avg


def adjust_lr(epoch, optimizer, base_lr):
    # TODO: Set it according to iCaRL
    if epoch < 80:
        lr = base_lr
    elif epoch < 120:
        lr = base_lr * 0.1
    else:
        lr = base_lr * 0.01
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def main():
    parser = argparse.ArgumentParser(description="SMILe: SubModular Incremental Learning")
    parser.add_argument("--repeat-rounds", default=1, type=int, help="The number of rounds the whole experiment needs"
                                                                     " to be performed.")
    parser.add_argument("--class-per-episode", default=100, type=int, help="Number of classes introduced per episode.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs each episode needs to be trained.")
    parser.add_argument("--batch-size", default=256, type=int, help="Size of each batch of datapoints for SGD.")
    parser.add_argument("--batch-size-test", default=100, type=int, help="Size of each batch of datapoints for SGD.")
    parser.add_argument("--learning-rate", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum parameter")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--gpu-ids", default='7', type=str, help="GPUs to be used for training.")
    parser.add_argument("--seed", default=99, type=int, help="The seed for randomness.")

    args = parser.parse_args()

    if not os.path.exists('result'):
        os.makedirs('result')

    if not os.path.exists('log'):
        os.makedirs('log')

    gpu_list = args.gpu_ids.split(',')
    gpus = [int(iter) for iter in gpu_list]
    torch.cuda.set_device(gpus[0])
    torch.backends.cudnn.benchmark = True

    if args.seed != 0:
        np.random.seed(args.seed)

    learn_incrementally(args, gpus)


if __name__ == "__main__":
    main()
