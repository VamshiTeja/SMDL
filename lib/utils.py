import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import logging, sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from lib.config import cfg


class Metrics:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


def plot_per_epoch_accuracies(train_accs, test_accs, episode_count, round_count):
    try:
        x = np.arange(1, len(train_accs)+1)
        plt.plot(x, train_accs, 'r', label='Train Accuracy')
        plt.plot(x, test_accs, 'g', label='Test Accuracy')
        plt.legend()
        plt.savefig(cfg.output_dir + '/plots/round_' + str(round_count) +'_episode_' + str(episode_count) + "_train_test_accuracy.png")
        plt.close()
    except Exception as error:
        log('Exception occurred while plotting the accuracies. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def plot_per_episode_accuracies(test_accs, round_count, num_classes):
    try:
        x = np.arange(0, num_classes+1)
        y = np.full_like(x, -10)
        step = num_classes / len(test_accs)
        for i in range(1, len(test_accs)+1):
            y[i * step] = test_accs[i-1]
        plt.scatter(x, y, color="blue", s=30, label="Test accuracy per episode")
        # plt.legend()
        plt.axis([0, num_classes + 10, 0, 100])
        plt.title('Accuracy in Round ' + str(round_count))
        plt.savefig(cfg.output_dir + '/plots/round_' + str(round_count) + "_episode_accuracy.png")
        plt.close()
    except Exception as error:
        log('Exception occurred while plotting the accuracies. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def plot_accuracies(test_accs, num_classes, color='blue', file_name_prefix='', title=None):
    step = num_classes / len(test_accs[0])
    x = range(step, num_classes+1, step)

    mean = np.mean(test_accs, axis=0)
    sd = np.std(test_accs, axis=0)

    y = mean
    err = sd
    plt.errorbar(x, y, yerr=err, color=color)
    plt.axis([0, num_classes+5, 0, 100+5])

    if title is not None:
        plt.title(title)
    if file_name_prefix != '':
        file_name_prefix = file_name_prefix + '_'
    plt.xlabel('Number of classes')
    plt.ylabel('Accuracy')
    plt.savefig(cfg.output_dir + '/plots/' + file_name_prefix + 'accuracy.png')
    plt.close()


def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print message


def save_accuracies(accuracy, file_name_prefix):
    file_name = file_name_prefix + '_accuracy.pkl'
    with open(file_name, 'w') as f:
        pickle.dump(accuracy, f)


if __name__ == '__main__':
    plot_accuracies([[80, 75, 32, 14], [90, 70, 20, 18], [60, 70, 30, 9]], 100, title='This is a title')
    save_accuracies([[9]], 'file')