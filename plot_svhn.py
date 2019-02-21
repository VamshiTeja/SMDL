import numpy as np
import matplotlib.pyplot as plt
import sys, os
from lib.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def plot_accuracies(data, title='Accuracy Plot', plot_type='Accuracy', x_axis_label='Epochs', save_location=None):
    """
    This helper function can be used to plot(visualize) the accuracies saved using lib.utils.save_accuracies()
    :return: None
    """
    if(save_location==None):
        save_location = './final_plots/svhn/'

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    for (i,info) in enumerate(data):
        with open(info[0], 'rb') as f:
            acc = pickle.load(f)
        _plot_indiv_accuracies(acc, color=info[2], label=info[1], plot_type=plot_type)
        if(i==0):
            with open(info[3], 'rb') as f:
                upper_limit = pickle.load(f)
            with open(info[4], 'rb') as f:
                lower_limit = pickle.load(f)
            x = np.arange(1, len(acc) + 1)
            if(plot_type=="Accuracy"):
                lower_limit = 100-lower_limit
                upper_limit = 100-upper_limit
            plt.fill_between(x, lower_limit, upper_limit)

    plt.legend()
    # if title is not None:
    #     plt.title(title)
    plt.xlabel(x_axis_label)

    plt.grid(True, linestyle='--', axis='y')

    # plt.xticks(np.arange(0, 101, step=20))

    if plot_type == 'Accuracy':
        plt.yticks(np.arange(0, 110, step=10))
        plt.ylabel('Error')
        plt.ylim([0,40])
    else:
        # plt.yticks(np.arange(0, 2, step=0.5))
        plt.ylabel('Loss')
        plt.ylim([0,1])

    plt.savefig(save_location +"/"+ title.replace(' ', '_').replace('(', '_').replace(')', '_') + '.eps', format='eps')
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


    ###################################################################################################################################
    # # SVHN

    # Main Result
    test_data = [
                    [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_acc_round_0_accuracy.pkl',
                          'Submodular Minibatch', 'green']
                      ]
    plot_accuracies(test_data, title='SVHN Test Error (Main)')

    train_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_round_0_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_lower_limit_accuracy.pkl' ],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_acc_round_0_accuracy.pkl',
                          'Submodular Minibatch', 'green']
                      ]
    plot_accuracies(train_data, title='SVHN Train Error (Main)')

    train_loss_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_upper_limit_accuracy.pkl',
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_loss_round_0_accuracy.pkl',
                          'Submodular Minibatch', 'green']

                      ]
    plot_accuracies(train_loss_data, title='SVHN Train Loss (Main)', plot_type='Loss')

    test_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_mean_accuracy.pkl',
        'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_loss_round_0_accuracy.pkl',
            'Submodular Minibatch', 'green']

    ]
    plot_accuracies(test_loss_data, title='SVHN Test Loss (Main)', plot_type='Loss')


##################################################################################################
    #SVHN Ablations

    ##################### refresh rate ###########################

    test_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_acc_round_0_accuracy.pkl',
                          'Submodular Minibatch Refresh Rate-5', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh10_0212_101539/accuracies/test_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-10', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh25_0212_101854/accuracies/test_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-25', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh50_0212_102202/accuracies/test_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-50', 'black']
                      ]
    plot_accuracies(test_data, title='SVHN Test Error (RF)', save_location='./final_plots/svhn/Refresh')

    train_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_round_0_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_acc_round_0_accuracy.pkl',
                          'Submodular Minibatch Refresh Rate-5', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh10_0212_101539/accuracies/train_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-10', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh25_0212_101854/accuracies/train_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-25', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh50_0212_102202/accuracies/train_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-50', 'black']
                      ]
    plot_accuracies(train_data, title='SVHN Train Accuracy (RF)', save_location='./final_plots/svhn/Refresh')

    train_loss_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_upper_limit_accuracy.pkl',
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_loss_round_0_accuracy.pkl',
                          'Submodular Minibatch Refresh Rate-5', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh10_0212_101539/accuracies/train_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-10', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh25_0212_101854/accuracies/train_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-25', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh50_0212_102202/accuracies/train_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-50', 'black'],
                      ]
    plot_accuracies(train_loss_data, title='SVHN Train Loss (RF)', plot_type='Train Loss', save_location='./final_plots/svhn/Refresh')

    test_loss_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_mean_accuracy.pkl',
        'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_loss_round_0_accuracy.pkl',
                          'Submodular Minibatch Refresh Rate-5', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh10_0212_101539/accuracies/test_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-10', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh25_0212_101854/accuracies/test_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-25', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_Refresh50_0212_102202/accuracies/test_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch Refresh Rate-50', 'black'],
                      ]
    plot_accuracies(test_loss_data, title='SVHN Test Loss (RF)', plot_type='Test Loss', save_location='./final_plots/svhn/Refresh')



    ################### batch_size ###########################

    test_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_acc_round_0_accuracy.pkl',
                          'Submodular Minibatch batch size-50', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch100_0215_225602/accuracies/test_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-100', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch200_0215_225815/accuracies/test_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-200', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch500_0215_230014/accuracies/test_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-500', 'black']
                      ]
    plot_accuracies(test_data, title='SVHN Test Error (BS)', save_location='./final_plots/svhn/Batch')

    train_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_round_0_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_acc_round_0_accuracy.pkl',
                          'Submodular Minibatch batch size-50', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch100_0215_225602/accuracies/train_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-100', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch200_0215_225815/accuracies/train_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-200', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch500_0215_230014/accuracies/train_acc_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-500', 'black']
                      ]
    plot_accuracies(train_data, title='SVHN Train Error (BS)', save_location='./final_plots/svhn/Batch')

    test_loss_data = [[
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_mean_accuracy.pkl',
        'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_lower_limit_accuracy.pkl'],
                      [
                          '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_loss_round_0_accuracy.pkl',
                          'Submodular Minibatch batch size-50', 'green'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch100_0215_225602/accuracies/test_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-100', 'orange'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch200_0215_225815/accuracies/test_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-200', 'm'],
                    [
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch500_0215_230014/accuracies/test_loss_round_0_accuracy.pkl',
                        'Submodular Minibatch batch size-500', 'black'],
                      ]
    plot_accuracies(test_loss_data, title='SVHN Test Loss (BS)', plot_type='Loss', save_location='./final_plots/svhn/Batch')

    train_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_upper_limit_accuracy.pkl',
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch batch size-50', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch100_0215_225602/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch batch size-100', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch200_0215_225815/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch batch size-200', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_batch500_0215_230014/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch batch size-500', 'black'],
    ]
    plot_accuracies(train_loss_data, title='SVHN Train Loss (BS)', plot_type='Loss', save_location='./final_plots/svhn/Batch')

    ######################################## learning rate #######################################

    test_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.01_0215_230546/accuracies/test_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.01', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.001_0215_230720/accuracies/test_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.001', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.0001_0215_230847/accuracies/test_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.0001', 'black']
    ]
    plot_accuracies(test_data, title='SVHN Test Error (LR)', save_location='./final_plots/svhn/LR')

    train_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_round_0_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.01_0215_230546/accuracies/train_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.01', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.001_0215_230720/accuracies/train_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.001', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.0001_0215_230847/accuracies/train_acc_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.0001', 'black']
    ]
    plot_accuracies(train_data, title='SVHN Train Error (LR)', save_location='./final_plots/svhn/LR')

    test_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_mean_accuracy.pkl',
        'SGD', 'blue',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/test_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.01_0215_230546/accuracies/test_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.01', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.001_0215_230720/accuracies/test_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.001', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.0001_0215_230847/accuracies/test_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.0001', 'black'],
    ]
    plot_accuracies(test_loss_data, title='SVHN Test Loss (LR)', plot_type='Loss',save_location='./final_plots/svhn/LR')

    train_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_upper_limit_accuracy.pkl',
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_0211_154707/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.01_0215_230546/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.01', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.001_0215_230720/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.001', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_lr0.0001_0215_230847/accuracies/train_loss_round_0_accuracy.pkl',
            'Submodular Minibatch lr-0.0001', 'black'],
    ]
    plot_accuracies(train_loss_data, title='SVHN Train Loss (LR)', plot_type='Loss',save_location='./final_plots/svhn/LR')


    ################################################ alphas #################################################
    test_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_upper_limit_accuracy.pkl',
                            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_acc_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha1_0215_194413/accuracies/test_acc_round_0_accuracy.pkl',
            'alpha1=1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha2_0215_194627/accuracies/test_acc_round_0_accuracy.pkl',
            'alpha2=1', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha3_0215_194850/accuracies/test_acc_round_0_accuracy.pkl',
            'alpha3=1', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha4_0215_195102/accuracies/test_acc_round_0_accuracy.pkl',
            'alpha4=1', 'black']
    ]
    plot_accuracies(test_data, title='SVHN Test Error (alphas)',save_location='./final_plots/svhn/alphas')

    train_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_round_0_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_acc_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha1_0215_194413/accuracies/train_acc_round_0_accuracy.pkl',
            'alpha1=1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha2_0215_194627/accuracies/train_acc_round_0_accuracy.pkl',
            'alpha2=1', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha3_0215_194850/accuracies/train_acc_round_0_accuracy.pkl',
            'alpha3=1', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha4_0215_195102/accuracies/train_acc_round_0_accuracy.pkl',
            'alpha4=1', 'black']
    ]
    plot_accuracies(train_data, title='SVHN Train Error (alphas)',save_location='./final_plots/svhn/alphas')

    test_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_mean_accuracy.pkl',
        'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_upper_limit_accuracy.pkl',
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/test_loss_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha1_0215_194413/accuracies/test_loss_round_0_accuracy.pkl',
            'alpha1=1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha2_0215_194627/accuracies/test_loss_round_0_accuracy.pkl',
            'alpha2=1', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha3_0215_194850/accuracies/test_loss_round_0_accuracy.pkl',
            'alpha3=1', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha4_0215_195102/accuracies/test_loss_round_0_accuracy.pkl',
            'alpha4=1', 'black'],
    ]
    plot_accuracies(test_loss_data, title='SVHN Test Loss (alphas)', plot_type='Loss',save_location='./final_plots/svhn/alphas')

    train_loss_data = [[
        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_mean_accuracy.pkl',
                          'SGD', 'blue', '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_upper_limit_accuracy.pkl',
                        '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SGD_SVHN_ResNet20_0211_151827/accuracies/train_loss_lower_limit_accuracy.pkl'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha1_0215_194413/accuracies/train_loss_round_0_accuracy.pkl',
            'alpha1=1', 'green'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha2_0215_194627/accuracies/train_loss_round_0_accuracy.pkl',
            'alpha2=1', 'orange'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha3_0215_194850/accuracies/train_loss_round_0_accuracy.pkl',
            'alpha3=1', 'm'],
        [
            '/home/vamshi/PycharmProjects/SMDL/final_Results/final_SMDL_SVHN_ResNet20_apha4_0215_195102/accuracies/train_loss_round_0_accuracy.pkl',
            'alpha4=1', 'black'],
    ]
    plot_accuracies(train_loss_data, title='SVHN Train Loss (alphas)', plot_type='Loss',save_location='./final_plots/svhn/alphas')
