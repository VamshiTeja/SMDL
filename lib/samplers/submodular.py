import numpy as np
import time
import copy
import scipy
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from scipy.spatial.distance import cdist
from torch.nn.functional import normalize
from torch import Tensor
import random
import torch
import torch.nn.functional as F

from lib.config import cfg
from sampler import Sampler
from lib.utils import log


class SubModSampler(Sampler):
    def __init__(self, model, dataset, batch_size, ltl_log_ep=5):
        super(SubModSampler, self).__init__(model, dataset)
        self.batch_size = batch_size
        self.index_set = range(0, len(self.dataset))    # It contains the indices of each image of the set.
        self.ltl_log_ep = ltl_log_ep

        f_acts = torch.tensor(self.final_activations)
        p_log_p = F.softmax(f_acts, dim=1) * F.log_softmax(f_acts, dim=1)
        H = -p_log_p.numpy()
        self.H = np.sum(H,axis=1)                       # Compute entropy of all samples for an epoch.
        self.penultimate_activations = np.array(self.penultimate_activations)

        penultimate_activations = torch.tensor(self.penultimate_activations)
        relu = torch.nn.ReLU(inplace=True)
        penultimate_activations = relu(penultimate_activations)

        softmax = torch.nn.Softmax()
        self.normalised_penultimate_activations = softmax(penultimate_activations).numpy()

        # col_sums = penultimate_activations.sum(axis=0)
        # self.normalised_penultimate_activations = penultimate_activations / col_sums[np.newaxis, :]

        # self.normalised_penultimate_activations = f_acts.numpy()

    def get_subset(self, detailed_logging=False):

        set_size = len(self.index_set)
        num_of_partitions = cfg.num_of_partitions

        if set_size >= num_of_partitions*self.batch_size:
            size_of_each_part = set_size / num_of_partitions
            r_size = (size_of_each_part*self.ltl_log_ep)/self.batch_size
            random.shuffle(self.index_set)
            partitions = [self.index_set[k:k+size_of_each_part] for k in range(0, set_size, size_of_each_part)]

            pool = ThreadPool(processes=len(partitions))
            pool_handlers = []
            for partition in partitions:
                handler = pool.apply_async(get_subset_indices, args=(partition, self.penultimate_activations, self.normalised_penultimate_activations,
                                                self.H, self.batch_size, r_size))
                pool_handlers.append(handler)
            pool.close()
            pool.join()

            intermediate_indices = []
            for handler in pool_handlers:
                intermediate_indices.extend(handler.get())
        else:
            intermediate_indices = self.index_set

        r_size = len(intermediate_indices) / self.batch_size * self.ltl_log_ep

        if detailed_logging:
            log('\nSelected {0} items from {1} partitions: {2} items.'.format(self.batch_size, num_of_partitions, len(intermediate_indices)))
            log('Size of random sample: {}'.format(r_size))

        subset_indices = get_subset_indices(intermediate_indices, self.penultimate_activations, self.normalised_penultimate_activations, self.H,
                                            self.batch_size, r_size)

        for item in subset_indices:     # Subset selection without replacement.
            self.index_set.remove(item)

        if detailed_logging:
            log('The selected {0} indices (second level): {1}'.format(len(subset_indices), subset_indices))
        return subset_indices


def get_subset_indices(index_set_input, penultimate_activations, normalised_penultimate_activations, entropy,  subset_size, r_size, alpha_1=cfg.alpha_1, alpha_2=cfg.alpha_2, alpha_3=cfg.alpha_3, alpha_4=cfg.alpha_4):

    if r_size < len(index_set_input):
        index_set = np.random.choice(index_set_input, r_size, replace=False)
    else:
        index_set = copy.deepcopy(index_set_input)

    subset_indices = []     # Subset of indices. Keeping track to improve computational performance.

    class_mean = np.mean(penultimate_activations, axis=0)

    subset_size = min(subset_size, len(index_set))
    for i in range(0, subset_size):
        now = time.time()
        # d_score = np.sum(compute_d_score(penultimate_activations,list(subset_indices)))
        # d_scores = d_score + compute_d_score(penultimate_activations, list(index_set))

        # u_score = np.sum(compute_u_score(entropy, list(subset_indices)))
        u_scores = compute_u_score(entropy, list(index_set), alpha=alpha_1)

        r_scores = compute_r_score(penultimate_activations, list(subset_indices), list(index_set), alpha=alpha_2)

        md_scores = compute_md_score(penultimate_activations, list(index_set), class_mean, alpha=alpha_3)

        coverage_scores = compute_coverage_score(normalised_penultimate_activations, subset_indices, index_set, alpha=alpha_4)

        scores = normalise(np.array(u_scores)) + normalise(np.array(r_scores)) + normalise(np.array(md_scores)) + normalise(np.array(coverage_scores))

        best_item_index = np.argmax(scores)
        subset_indices.append(index_set[best_item_index])
        index_set = np.delete(index_set, best_item_index, axis=0)

        # log('Processed: {0}/{1} exemplars. Time taken is {2} sec.'.format(i, subset_size, time.time()-now))

    return subset_indices

def normalise(A):
    return A/np.sum(A)

# def compute_d_score(penultimate_activations, subset_indices, alpha=1.):
#     """
#     Computes the Diversity Score: The new point should be distant from all the elements in the subset.
#     :param penultimate_activations:
#     :param subset_indices:
#     :param alpha:
#     :return: d_score
#     """
#     if len(subset_indices) <= 1:
#         return 0
#     else:
#         p_acts = itemgetter(*subset_indices)(penultimate_activations)
#         pdist = cdist(p_acts, penultimate_activations, metric='cosine')
#         return np.sum(pdist, axis=1)


def compute_u_score(entropy, index_set, alpha=1.):
    """
    Compute the Uncertainity Score: The point that makes the model most confused, should be preferred.
    :param final_activations:
    :param subset_indices:
    :param alpha:
    :return: u_score
    """

    if len(index_set) == 0:
        return 0
    else:
        u_score = alpha*entropy[index_set]
        return u_score


def compute_r_score(penultimate_activations, subset_indices, index_set, alpha=0.2, distance_metric=cfg.distance_metric):
    """
    Computes Redundancy Score: The point should be distant from all the other elements in the subset.
    :param penultimate_activations:
    :param subset_indices:
    :param alpha:
    :return:
    """
    if len(subset_indices) == 0:
        return 0
    else:
        index_p_acts = penultimate_activations[np.array(index_set)]
        subset_p_acts = penultimate_activations[np.array(subset_indices)]
        if(distance_metric=='gaussian'):
            pdist = cdist(index_p_acts, subset_p_acts, metric='sqeuclidean')
            r_score = scipy.exp(-pdist / (0.5) ** 2)
            r_score = alpha * np.min(r_score, axis=1)
            return r_score
        else:
            pdist = cdist(index_p_acts, subset_p_acts, metric=distance_metric)
            r_score = alpha * np.min(pdist, axis=1)
            return r_score


def compute_md_score(penultimate_activations, index_set, class_mean, alpha=0.2, distance_metric=cfg.distance_metric):
    """
    Computes Mean Divergence score: The new datapoint should be close to the class mean
    :param penultimate_activations:
    :param index_set:
    :param class_mean:
    :param alpha:
    :return: list of scores for each index item
    """

    if(distance_metric=='gaussian'):
        pen_act = penultimate_activations[np.array(index_set)]
        md_score = alpha * cdist(pen_act, np.array([np.array(class_mean)]), metric='sqeuclidean')
        md_score = scipy.exp(-md_score / (0.5) ** 2)
        return md_score
    else:
        pen_act = penultimate_activations[np.array(index_set)]
        md_score = alpha * cdist(pen_act, np.array([np.array(class_mean)]), metric=distance_metric)
        md_score = np.squeeze(md_score, 1)
        return -md_score


def compute_coverage_score(normalised_penultimate_activations, subset_indices, index_set, alpha=0.5):
    """
    :param penultimate_activations:
    :param subset_indices:
    :param index_set:
    :return: g(mu(S))
    """
    if(len(subset_indices)==0):
        penultimate_activations_index_set = normalised_penultimate_activations[index_set]
        score_feature_wise = np.sqrt(penultimate_activations_index_set)
        scores = np.sum(score_feature_wise, axis=1)
        return alpha*scores
    else:
        penultimate_activations_index_set =  normalised_penultimate_activations[index_set]
        subset_indices_scores = np.sum(normalised_penultimate_activations[subset_indices],axis=0)
        sum_subset_index_set = subset_indices_scores + penultimate_activations_index_set
        score_feature_wise = np.sqrt(sum_subset_index_set)
        scores = np.sum(score_feature_wise,axis=1)
        return alpha*scores