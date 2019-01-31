import numpy as np
import matplotlib.pyplot as plt
import sys
from lib.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def plot_accuracies(data, title='Accuracy Plot', plot_type='Accuracy', x_axis_label='Epochs'):
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
    plt.xlabel(x_axis_label)

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
    #
    for i in range(1, 41):
        try:
            test_data = [[
                                  '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_random_full_0119_214903/accuracies/test_acc_between_iteration_epoch_' + str(i) + '_accuracy.pkl',
                                  'Random Selection', 'blue'],
                              [
                                  '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_submodcomb_latest_0131_010429/accuracies/test_acc_between_iteration_epoch_' + str(i) + '_accuracy.pkl',
                                  'Submodular Selection', 'green']
                              ]
            plot_accuracies(test_data, title='SVHN Epoch ' + str(i) + ' Test Accuracy', x_axis_label='# of iterations (x10)')
        except Exception as error:
            print ('Exception occured for index {}, {}'.format(i, error))

    # CIFAR 10
    # test_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar10_resnet20_random_full_0119_214639/accuracies/test_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar10_resnet20_submod_comb_0128_211805/accuracies/test_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(test_data, title='CIFAR 10 (Early Random Stopping) Test Accuracy')
    #
    # train_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar10_resnet20_random_full_0119_214639/accuracies/train_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar10_resnet20_submod_comb_0128_211805/accuracies/train_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(train_data, title='CIFAR 10 (with warm-up) Train Accuracy')
    #
    # loss_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar10_resnet20_random_full_0119_214639/accuracies/loss_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar10_resnet20_submod_comb_0128_211805/accuracies/loss_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(loss_data, title='CIFAR 10 (with warm-up) Loss', plot_type='Loss')

    # CIFAR - 100
    #-------------------
    # test_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar100_resnet32_random_full_0119_214225/accuracies/test_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar100_resnet32_submod_new_0125_220927/accuracies/test_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(test_data, title='CIFAR 100 Test Accuracy')
    #
    # train_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar100_resnet32_random_full_0119_214225/accuracies/train_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar100_resnet32_submod_new_0125_220927/accuracies/train_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(train_data, title='CIFAR 100 Train Accuracy')
    #
    # loss_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar100_resnet32_random_full_0119_214225/accuracies/loss_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/cifar100_resnet32_submod_new_0125_220927/accuracies/loss_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(loss_data, title='CIFAR 100 Loss', plot_type='Loss')

    # SVHN
    test_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_random_full_0119_214903/accuracies/test_acc_round_0_accuracy.pkl',
                          'Random Selection', 'blue'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_submodcomb_latest_0131_010429/accuracies/test_acc_round_0_accuracy.pkl',
                          'Submodular Selection', 'green']
                      ]
    plot_accuracies(test_data, title='SVHN Test Accuracy')

    train_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_random_full_0119_214903/accuracies/train_acc_round_0_accuracy.pkl',
                          'Random Selection', 'blue'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_submodcomb_latest_0131_010429/accuracies/train_acc_round_0_accuracy.pkl',
                          'Submodular Selection', 'green']
                      ]
    plot_accuracies(train_data, title='SVHN Train Accuracy')

    loss_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_random_full_0119_214903/accuracies/loss_round_0_accuracy.pkl',
                          'Random Selection', 'blue'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/output/svhn_resnet20_submodcomb_latest_0131_010429/accuracies/loss_round_0_accuracy.pkl',
                          'Submodular Selection', 'green']
                      ]
    plot_accuracies(loss_data, title='SVHN Loss', plot_type='Loss')

    # F-MNIST
    # test_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/fmnist_simplenet_random_full_0119_220306/accuracies/test_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/fmnist_simplenet_submod_new_0125_221439/accuracies/test_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(test_data, title='F-MNIST Test Accuracy')
    #
    # train_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/fmnist_simplenet_random_full_0119_220306/accuracies/train_acc_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/fmnist_simplenet_submod_new_0125_221439/accuracies/train_acc_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(train_data, title='F-MNIST Train Accuracy')
    #
    # loss_data = [[
    #                       '/home/vamshi/PycharmProjects/SMDL/output/fmnist_simplenet_random_full_0119_220306/accuracies/loss_round_0_accuracy.pkl',
    #                       'Random Selection', 'blue'],
    #                   [
    #                       '/home/vamshi/PycharmProjects/SMDL/output/fmnist_simplenet_submod_new_0125_221439/accuracies/loss_round_0_accuracy.pkl',
    #                       'Submodular Selection', 'green']
    #                   ]
    # plot_accuracies(loss_data, title='F-MNIST Loss', plot_type='Loss')