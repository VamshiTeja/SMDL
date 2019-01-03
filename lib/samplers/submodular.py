import numpy as np
import time
import copy
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from scipy.spatial.distance import cdist
from torch.nn.functional import normalize
from torch import Tensor

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
        self.dist = [1./len(self.dataset)]*len(self.dataset)

    def get_subset(self, detailed_logging=False):

        set_size = len(self.index_set)
        num_of_partitions = cfg.num_of_partitions

        if set_size >= num_of_partitions*self.batch_size:
            size_of_each_part = set_size / num_of_partitions
            r_size = (size_of_each_part*self.ltl_log_ep)/self.batch_size
            partitions = [self.index_set[k:k+size_of_each_part] for k in range(0, set_size, size_of_each_part)]

            pool = ThreadPool(processes=len(partitions))
            pool_handlers = []
            for partition in partitions:
                handler = pool.apply_async(get_subset_indices, args=(partition, self.penultimate_activations, self.final_activations,
                                                self.H, self.batch_size, r_size, self.dist))
                pool_handlers.append(handler)
            pool.close()
            pool.join()

            intermediate_indices = []
            self.dist = []
            for handler in pool_handlers:
                intermediate_indices.extend(handler.get()[0])
                self.dist.append(handler.get()[1])
        else:
            intermediate_indices = self.index_set

        r_size = len(intermediate_indices) / self.batch_size * self.ltl_log_ep

        if detailed_logging:
            log('\nSelected {0} items from {1} partitions: {2} items.'.format(self.batch_size, num_of_partitions, len(intermediate_indices)))
            log('Size of random sample: {}'.format(r_size))

        subset_indices, dist2level = get_subset_indices(intermediate_indices, self.penultimate_activations, self.final_activations, self.H,
                                            self.batch_size, r_size, self.dist)

        #Update distribution at 2nd level based on adaboost
        self.dist[intermediate_indices] = dist2level
        self.dist = self.dist/np.sum(self.dist)

        # Subset selection without replacement.
        #for item in subset_indices:
        #    self.index_set.remove(item)

        if detailed_logging:
            log('The selected {0} indices (second level): {1}'.format(len(subset_indices), subset_indices))
        return subset_indices


def get_subset_indices(index_set_input, penultimate_activations, final_activations, entropy,  subset_size, r_size, dist, alpha_1=1., alpha_2=1., alpha_3=1., alpha_4=0.5):

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
        u_scores = compute_u_score(entropy, list(index_set))

        r_scores = compute_r_score(penultimate_activations, list(subset_indices), list(index_set))

        md_scores = compute_md_score(penultimate_activations, list(index_set), class_mean)

        scores = normalise(np.array(u_scores)) + normalise(np.array(r_scores)) + normalise(np.array(md_scores)) + alpha_4*dist[index_set]

        best_item_index = np.argmax(scores)
        subset_indices.append(index_set[best_item_index])
        index_set = np.delete(index_set, best_item_index, axis=0)

        # Update distribution based on adaboost
        dist[best_item_index] = dist[best_item_index]*np.exp(-scores[best_item_index])
        dist = dist/np.sum(dist)

        # log('Processed: {0}/{1} exemplars. Time taken is {2} sec.'.format(i, subset_size, time.time()-now))

    return subset_indices, dist[index_set_input]

def normalise(A):
    std = np.std(A)
    if std==0:
        std = 1
    A = (A-np.mean(A))/std
    return A

def update_dist(dist_arr, submod_score):
    dist_arr = np.dot(dist_arr, np.exp(-submod_score))
    dist_arr = dist_arr/np.sum(dist_arr)
    return  dist_arr

def compute_d_score(penultimate_activations, subset_indices, alpha=1.):
    """
    Computes the Diversity Score: The new point should be distant from all the elements in the subset.
    :param penultimate_activations:
    :param subset_indices:
    :param alpha:
    :return: d_score
    """
    if len(subset_indices) <= 1:
        return 0
    else:
        p_acts = itemgetter(*subset_indices)(penultimate_activations)
        pdist = cdist(p_acts, penultimate_activations)
        return np.sum(pdist, axis=1)


def compute_u_score(entropy, subset_indices, alpha=1.):
    """
    Compute the Uncertainity Score: The point that makes the model most confused, should be preferred.
    :param final_activations:
    :param subset_indices:
    :param alpha:
    :return: u_score
    """

    if len(subset_indices) == 0:
        return 0
    else:
        u_score = alpha*entropy[subset_indices]
        return u_score


def compute_r_score(penultimate_activations, subset_indices, index_set, alpha=0.2):
    """
    Computes Redundancy Score: The point should be distant from all the other elements in the subset.
    :param penultimate_activations:
    :param subset_indices:
    :param alpha:
    :return:
    """
    if len(subset_indices) == 0:
        return 0
    elif len(index_set) == 0:
        return 0
    elif len(subset_indices) == 1:
        return [np.linalg.norm(np.array(itemgetter(*index_set)(penultimate_activations))-np.array(subset_indices[0]))]
    elif len(index_set) == 1:
        return [np.min(np.linalg.norm(np.array(penultimate_activations[index_set[0]])-np.array((itemgetter(*subset_indices)(penultimate_activations)))))]
    else:
        index_p_acts = np.array(itemgetter(*index_set)(penultimate_activations))
        subset_p_acts = np.array((itemgetter(*subset_indices)(penultimate_activations)))
        pdist = cdist(index_p_acts, subset_p_acts)
        r_score = alpha * np.min(pdist, axis=1)
        return r_score

    # r_score = 0
    # if len(subset_indices) > 1:
    #     subset_penultimate_acts = itemgetter(*subset_indices)(penultimate_activations)
    # for index in subset_indices:
    #     p_act = penultimate_activations[index]
    #     if len(subset_indices) > 1:
    #         dist = np.linalg.norm(subset_penultimate_acts - p_act, axis=1)
    #         score = np.min(dist[dist != 0])
    #         r_score += alpha * score
    # return r_score


def compute_md_score(penultimate_activations, index_set, class_mean, alpha=2.):
    """
    Computes Mean Divergence score: The new datapoint should be close to the class mean
    :param penultimate_activations:
    :param index_set:
    :param class_mean:
    :param alpha:
    :return: list of scores for each index item
    """
    if len(index_set) == 1:
        return np.linalg.norm(penultimate_activations[index_set[0]]-class_mean)
    else:
        pen_act = np.array(itemgetter(*index_set)(penultimate_activations)) - np.array(class_mean)
        md_score = alpha * np.linalg.norm(pen_act, axis=1)
        return -md_score
