import matplotlib.pyplot as plt
import numpy as np
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
        plt.plot(x, train_accs, 'r', x, test_accs, 'g')
        plt.savefig('./results/round_' + str(round_count) +'_episode_' + str(episode_count) + "_train_test_accuracy.png")
    except:
        print 'Exception occured while plotting the accuracies. Ignoring.'


if __name__ == "__main__":
    plot_per_epoch_accuracies([1,2,3],[4,5,6], 1, 1)