import numpy as np
import matplotlib.pyplot as plt
import sys
from lib.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def plot_accuracies(data, title='Accuracy Plot', plot_type='Accuracy'):
    """
    This helper function can be used to plot(visualize) the accuracies saved using lib.utils.save_accuracies()
    :return: None
    """
    save_location = './output/'

    for info in data:
        with open(info[0], 'rb') as f:
            acc = pickle.load(f)
        _plot_indiv_accuracies(acc, color=info[2], label=info[1])

    plt.legend()
    if title is not None:
        plt.title(title)
    plt.xlabel('Epochs')

    plt.grid(True, linestyle='--', axis='y')

    # plt.xticks(np.arange(0, 101, step=20))

    if plot_type == 'Accuracy':
        plt.yticks(np.arange(0, 110, step=10))
        plt.ylabel('Accuracy')
    else:
        # plt.yticks(np.arange(0, 2, step=0.5))
        plt.ylabel('Loss')

    plt.savefig(save_location + title.replace(' ', '_').replace('(', '_').replace(')', '_') + '.png')
    plt.close()


def _plot_indiv_accuracies(accuracies, color='blue', label=''):
    x = np.arange(1, len(accuracies) + 1)
    plt.plot(x, accuracies, color=color, label=label)


if __name__ == '__main__':

    test_data = [[
                          '/home/joseph/workspace/SMDL/output/Random_CIFAR_10_ResNet20_with_warmup_1125_203228/accuracies/test_acc_round_0_accuracy.pkl',
                          'Random Selection', 'blue'],
                      [
                          '/home/joseph/workspace/SMDL/output/SMDL_CIFAR_10_ResNet20_with_warmup_1125_203442/accuracies/test_acc_round_0_accuracy.pkl',
                          'Submodular Selection', 'green']
                      ]
    plot_accuracies(test_data, title='CIFAR 10 (with warm-up) Test Accuracy')

    train_data = [[
                          '/home/joseph/workspace/SMDL/output/Random_CIFAR_10_ResNet20_with_warmup_1125_203228/accuracies/train_acc_round_0_accuracy.pkl',
                          'Random Selection', 'blue'],
                      [
                          '/home/joseph/workspace/SMDL/output/SMDL_CIFAR_10_ResNet20_with_warmup_1125_203442/accuracies/train_acc_round_0_accuracy.pkl',
                          'Submodular Selection', 'green']
                      ]
    plot_accuracies(train_data, title='CIFAR 10 (with warm-up) Train Accuracy')

    loss_data = [[
                          '/home/joseph/workspace/SMDL/output/Random_CIFAR_10_ResNet20_with_warmup_1125_203228/accuracies/loss_round_0_accuracy.pkl',
                          'Random Selection', 'blue'],
                      [
                          '/home/joseph/workspace/SMDL/output/SMDL_CIFAR_10_ResNet20_with_warmup_1125_203442/accuracies/loss_round_0_accuracy.pkl',
                          'Submodular Selection', 'green']
                      ]
    plot_accuracies(loss_data, title='CIFAR 10 (with warm-up) Loss', plot_type='Loss')

    # CIFAR - 10 TEMPLATE
    # -------------------
    # test_data = [[
    #                       '/home/joseph/workspace/SMDL/output/Random_CIFAR10_ResNet20_1119_145735/accuracies/Random_CIFAR10_ResNet20_test_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/joseph/workspace/SMDL/output/SMDL_CIFAR10_ResNet20_1119_145915/accuracies/SMDL_CIFAR10_ResNet20_test_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(test_data, title='CIFAR 10 Test Accuracy')
    #
    # train_data = [[
    #                       '/home/joseph/workspace/SMDL/output/Random_CIFAR10_ResNet20_1119_145735/accuracies/Random_CIFAR10_ResNet20_train_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/joseph/workspace/SMDL/output/SMDL_CIFAR10_ResNet20_1119_145915/accuracies/SMDL_CIFAR10_ResNet20_train_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(train_data, title='CIFAR 10 Train Accuracy')
    #
    # loss_data = [[
    #                       '/home/joseph/workspace/SMDL/output/Random_CIFAR10_ResNet20_1119_145735/accuracies/Random_CIFAR10_ResNet20_loss_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/joseph/workspace/SMDL/output/SMDL_CIFAR10_ResNet20_1119_145915/accuracies/SMDL_CIFAR10_ResNet20_loss_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(loss_data, title='CIFAR 10 Loss', plot_type='Loss')

    # SVHN
    # test_data = [[
    #                       '/home/joseph/workspace/SMDL/output/Random_SVHN_ResNet20_1119_150132/accuracies/Random_SVHN_ResNet20_test_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/joseph/workspace/SMDL/output/output_dgx/SMDL_SVHN_ResNet20_1119_150345/accuracies/SMDL_SVHN_ResNet20_test_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(test_data, title='SVHN Test Accuracy')
    #
    # train_data = [[
    #                       '/home/joseph/workspace/SMDL/output/Random_SVHN_ResNet20_1119_150132/accuracies/Random_SVHN_ResNet20_train_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/joseph/workspace/SMDL/output/output_dgx/SMDL_SVHN_ResNet20_1119_150345/accuracies/SMDL_SVHN_ResNet20_train_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(train_data, title='SVHN Train Accuracy')
    #
    # loss_data = [[
    #                       '/home/joseph/workspace/SMDL/output/Random_SVHN_ResNet20_1119_150132/accuracies/Random_SVHN_ResNet20_loss_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/joseph/workspace/SMDL/output/output_dgx/SMDL_SVHN_ResNet20_1119_150345/accuracies/SMDL_SVHN_ResNet20_loss_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(loss_data, title='SVHN Loss', plot_type='Loss')