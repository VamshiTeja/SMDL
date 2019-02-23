import numpy as np
import matplotlib.pyplot as plt
import sys, os
from lib.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def plot_accuracies(data, title='Accuracy Plot', plot_type='Accuracy', x_axis_label='Epochs',save_location=None, mode='Test'):
    """
    This helper function can be used to plot(visualize) the accuracies saved using lib.utils.save_accuracies()
    :return: None
    """
    if(save_location==None):
        save_location = './final_plots/cifar100/'

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    for (i,info) in enumerate(data):
        with open(info[0], 'rb') as f:
            acc = pickle.load(f)
        _plot_indiv_accuracies(acc, color=info[2], label=info[1], plot_type=plot_type)
        if(info[1]=='SGD'):
            with open(info[3], 'rb') as f:
                upper_limit = pickle.load(f)
            with open(info[4], 'rb') as f:
                lower_limit = pickle.load(f)
            x = np.arange(1, len(acc) + 1)
            if(plot_type=="Accuracy"):
                lower_limit = 100-lower_limit
                upper_limit = 100-upper_limit
            plt.fill_between(x, lower_limit, upper_limit, color='lightskyblue')


    size = 12

    plt.legend(fontsize=size)
    # if title is not None:
    #     plt.title(title)
    plt.xlabel(x_axis_label,fontsize=size)

    plt.grid(True, linestyle='--', axis='y')

    # plt.xticks(np.arange(0, 101, step=20))

    if plot_type == 'Accuracy':
        plt.yticks(np.arange(0, 110, step=10))
        if (mode == 'Test'):
            plt.ylabel('Test Error',fontsize=size)
        elif (mode == 'Train'):
            plt.ylabel('Train Error',fontsize=size)

        plt.ylim([30,100])
    else:
        # plt.yticks(np.arange(0, 2, step=0.5))
        if (mode == 'Test'):
            plt.ylabel('Test Loss',fontsize=size)
        elif (mode == 'Train'):
            plt.ylabel('Train Loss',fontsize=size)

        plt.ylim([1,4])

    plt.savefig(save_location + title.replace(' ', '_').replace('(', '_').replace(')', '_') + '.eps', format='eps')
    plt.close()


def _plot_indiv_accuracies(accuracies, color='blue', label='', plot_type=None):
    x = np.arange(1, len(accuracies) + 1)
    if(plot_type=='Accuracy'):
        accuracies = 100 - np.array(accuracies)
    plt.plot(x, accuracies, color=color, label=label)


if __name__ == '__main__':
    #
    # for i in range(1, 61):
    #     try:
    #         test_data = [[
    #                               '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_ResNet20_0211_152357/accuracies/test_acc_between_iteration_epoch_' + str(i) + '_accuracy.pkl',
    #                               'SGD', 'blue'],
    #                           [
    #                               '/home/vamshi/PycharmProjects/SMDL/output/cifar100_resnet32_submodcomb_refresh-5_epochs-60_0203_103925/accuracies/test_acc_between_iteration_epoch_' + str(i) + '_accuracy.pkl',
    #                               'Submodular Selection', 'green']
    #                           ]
    #         plot_accuracies(test_data, title='CIFAR100 Epoch ' + str(i) + ' Test Accuracy', x_axis_label='# of iterations (x10)')
    #     except Exception as error:
    #         print ('Exception occured for index {}, {}'.format(i, error))



    # # CIFAR - 100
    # -------------------
    test_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_acc_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_acc_lower_limit_accuracy.pkl'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_acc_round_0_accuracy.pkl',
                        'LOSS', 'darkviolet'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/test_acc_round_0_accuracy.pkl',
                          'SMDL', 'green']
                      ]
    plot_accuracies(test_data, title='CIFAR 100 Test Accuracy (Main)')

    train_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_acc_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_acc_lower_limit_accuracy.pkl'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_acc_round_0_accuracy.pkl',
                        'LOSS', 'darkviolet'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/train_acc_round_0_accuracy.pkl',
                          'SMDL', 'green']
                      ]
    plot_accuracies(train_data, title='CIFAR 100 Train Accuracy (Main)')

    test_loss_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_lower_limit_accuracy.pkl'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_round_0_accuracy.pkl',
                        'LOSS', 'darkviolet'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/test_loss_round_0_accuracy.pkl',
                          'SMDL', 'green']
                      ]
    plot_accuracies(test_loss_data, title='CIFAR 100 Test Loss (Main)', plot_type='Loss')

    train_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_loss_mean_accuracy.pkl',
        'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_loss_lower_limit_accuracy.pkl'],
        [
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_round_0_accuracy.pkl',
        'LOSS', 'darkviolet'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/test_loss_round_0_accuracy.pkl',
            'SMDL', 'green']
    ]
    plot_accuracies(train_loss_data, title='CIFAR 100 Train Loss (Main)', plot_type='Loss')


############## ablations ###############

########## refresh rate ##############

    test_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_acc_mean_accuracy.pkl',
                          'Random Selection', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_acc_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/test_acc_round_0_accuracy.pkl',
                          'SMDL Refresh Rate-5', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh10_0212_123549/accuracies/test_acc_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-10', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh25_0212_123328/accuracies/test_acc_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-25', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh50_0212_123140/accuracies/test_acc_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-50', 'black']
                      ]
    plot_accuracies(test_data, title='CIFAR 100 Test Error with RF Ablation', save_location='./final_plots/cifar100/Refresh/')

    train_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_acc_mean_accuracy.pkl',
                          'Random Selection', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_acc_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/train_acc_round_0_accuracy.pkl',
                          'SMDL Refresh Rate-5', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh10_0212_123549/accuracies/train_acc_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-10', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh25_0212_123328/accuracies/train_acc_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-25', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh50_0212_123140/accuracies/train_acc_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-50', 'black']
                      ]
    plot_accuracies(train_data, title='CIFAR 100 Train Error with RF Ablation', save_location='./final_plots/cifar100/Refresh/')

    test_loss_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_mean_accuracy.pkl',
                          'Random Selection', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/test_loss_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/test_loss_round_0_accuracy.pkl',
                          'SMDL Refresh Rate-5', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh10_0212_123549/accuracies/test_loss_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-10', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh25_0212_123328/accuracies/test_loss_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-25', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh50_0212_123140/accuracies/test_loss_round_0_accuracy.pkl',
                        'SMDL Refresh Rate-50', 'black']
                      ]
    plot_accuracies(test_loss_data, title='CIFAR 100 Test Loss with RF Ablation', plot_type='Loss', save_location='./final_plots/cifar100/Refresh/')

    train_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_loss_mean_accuracy.pkl',
        'Random Selection', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_CIFAR_100_ResNet32_0211_152526/accuracies/train_loss_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/fix_SMDL_CIFAR_100_ResNet32_0215_182025/accuracies/test_loss_round_0_accuracy.pkl',
            'SMDL Refresh Rate-5', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh10_0212_123549/accuracies/test_loss_round_0_accuracy.pkl',
            'SMDL Refresh Rate-10', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh25_0212_123328/accuracies/test_loss_round_0_accuracy.pkl',
            'SMDL Refresh Rate-25', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_CIFAR_100_ResNet32_Refresh50_0212_123140/accuracies/test_loss_round_0_accuracy.pkl',
            'SMDL Refresh Rate-50', 'black']
    ]
    plot_accuracies(train_loss_data, title='CIFAR 100 Train Loss with RF Ablation', plot_type='Loss', save_location='./final_plots/cifar100/Refresh/')