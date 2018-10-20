import numpy as np
import matplotlib.pyplot as plt
import sys
from lib.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def plot_accuracies():
    """
    This helper function can be used to plot(visualize) the accuracies saved using lib.utils.save_accuracies()
    :return: None
    """
    accuracy_infos = [['/home/joseph/il/smile/output/smile_1018_120912/accuracies/smile_accuracy.pkl', 'Using randomly selected exemplar set', 'green'],
                      ['/home/joseph/il/smile/output/smile_all_data_1018_122812/accuracies/smile_all_data_accuracy.pkl', 'Using all the data', 'blue'],
                      ['/home/joseph/il/smile/output/submodular_smile_1_round_1020_105605/accuracies/submodular_smile_1_round_accuracy.pkl', 'Submodular subset selection', 'violet']]

    save_location = './'
    title = ''

    for info in accuracy_infos:
        with open(info[0], 'rb') as f:
            acc = pickle.load(f)
        _plot_indiv_accuracies(acc, color=info[2], label=info[1])

    plt.legend()
    if title is not None:
        plt.title(title)
    plt.xlabel('Number of classes')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', axis='y')
    plt.xticks(np.arange(0, 101, step=20))
    plt.yticks(np.arange(0, 101, step=10))

    plt.savefig(save_location + 'combined_accuracy.png')
    plt.close()


def _plot_indiv_accuracies(test_accs, color='blue', label=''):
    num_classes = cfg.dataset.total_num_classes
    step = num_classes / len(test_accs[0])
    x = range(step, num_classes+1, step)

    mean = np.mean(test_accs, axis=0)
    sd = np.std(test_accs, axis=0)

    y = mean
    err = sd
    plt.errorbar(x, y, yerr=err, color=color, label=label)
    plt.axis([0, num_classes+5, 0, 100+5])


if __name__ == '__main__':
    plot_accuracies()
