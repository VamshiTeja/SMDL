import numpy as np
import time
import copy
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from scipy.spatial.distance import cdist

import torch
import torch.nn.functional as F

from lib.config import cfg
from sampler import Sampler
from lib.utils import log


class SubModSampler(Sampler):
    def __init__(self, model, dataset, batch_size, r_size=2048):
        super(SubModSampler, self).__init__(model, dataset)
        self.batch_size = batch_size
        self.index_set = range(0, len(self.dataset))    # It contains the indices of each image of the set.
        self.r_size = r_size

    def get_subset(self):

        set_size = len(self.index_set)
        num_of_partitions = cfg.num_of_partitions

        if set_size >= num_of_partitions*self.batch_size:
            size_of_each_part = set_size / num_of_partitions
            partitions = [self.index_set[k:k+size_of_each_part] for k in range(0, set_size, size_of_each_part)]

            pool = ThreadPool(processes=len(partitions))
            pool_handlers = []
            for partition in partitions:
                handler = pool.apply_async(get_subset_indices, args=(partition, self.penultimate_activations, self.final_activations,
                                                self.batch_size, self.r_size))
                pool_handlers.append(handler)
            pool.close()
            pool.join()

            intermediate_indices = []
            for handler in pool_handlers:
                intermediate_indices.extend(handler.get())
        else:
            intermediate_indices = self.index_set

        log('Selected {0} items from {1} partitions: {2} items.'.format(self.batch_size, num_of_partitions, len(intermediate_indices)))

        subset_indices = get_subset_indices(intermediate_indices, self.penultimate_activations, self.final_activations,
                                            self.batch_size, self.r_size)

        for item in subset_indices:     # Subset selection without replacement.
            self.index_set.remove(item)

        log('The selected {0} indices (second level): {1}'.format(len(subset_indices), subset_indices))
        return subset_indices


def get_subset_indices(index_set_input, penultimate_activations, final_activations, subset_size, r_size, alpha_1=1, alpha_2=1, alpha_3=1):
    if(r_size<len(index_set_input)):
        index_set = np.random.choice(index_set_input,r_size,replace=False)
    else:
        index_set = copy.deepcopy(index_set_input)
    #index_set = copy.deepcopy(index_set_input)
    subset_indices = []     # Subset of indices. Keeping track to improve computational performance.

    class_mean = np.mean(penultimate_activations, axis=0)

    subset_size = min(subset_size, len(index_set))
    for i in range(0, subset_size):
        scores = []
        now = time.time()

        # Compute d_score for the whole subset. Then add the d_score of just the
        # new item to compute the total d_score.

        # Same logic for u_score

        # Same logic for md_score

        md_score = np.sum(compute_md_score(penultimate_activations, list(subset_indices), class_mean))
        md_scores = md_score+compute_md_score(penultimate_activations, list(index_set), class_mean)

        u_score = np.sum(compute_u_score(final_activations, list(subset_indices)))
        u_scores = u_score + compute_u_score(final_activations, list(index_set))

        #d_score = np.sum(compute_d_score(penultimate_activations,list(subset_indices)))
        #d_scores = d_score + compute_d_score(penultimate_activations, list(index_set))

        #r_scores = compute_r_score(penultimate_activations, list(subset_indices))

        scores = md_scores + u_scores

        '''
        for iter, item in enumerate(index_set):
            temp_subset_indices = list(subset_indices)
            temp_subset_indices.append(index_set[iter])

            #d_score += compute_d_score(penultimate_activations, list([index_set[iter]]))
            # u_score += compute_u_score(final_activations, list([index_set[iter]]))
            md_score += compute_md_score(penultimate_activations, (list([index_set[iter]])), class_mean)
            # r_score = compute_r_score(penultimate_activations, list(temp_subset_indices))

            #score = d_score + u_score + r_score + md_score
            score = md_score
            scores.append(score)
        '''

        best_item_index = np.argmax(scores)
        subset_indices.append(index_set[best_item_index])
        index_set = np.delete(index_set, best_item_index, axis=0)

        #log('Processed: {0}/{1} exemplars. Time taken is {2} sec.'.format(i, subset_size, time.time()-now))

    return subset_indices


def compute_d_score(penultimate_activations, subset_indices, alpha=1.):
    """
    Computes the Diversity Score: The new point should be distant from all the elements in the subset.
    :param penultimate_activations:
    :param subset_indices:
    :param alpha:
    :return: d_score
    """

    if(len(subset_indices)==0):
        return 0
    elif(len(subset_indices)==1):
        return 0
    else:
        p_acts = itemgetter(*subset_indices)(penultimate_activations)
        pdist = cdist(p_acts,penultimate_activations)
        return np.sum(pdist,axis=1)

    '''
    d_score = 0
    for index in subset_indices:
        p_act = penultimate_activations[index]
        all_acts = penultimate_activations
        score = np.sum(np.linalg.norm(all_acts - p_act, axis=1))
        d_score += alpha * score

    return d_score'''


def compute_u_score(final_activations, subset_indices, alpha=1.):
    """
    Compute the Uncertainity Score: The point that makes the model most confused, should be preferred.
    :param final_activations:
    :param subset_indices:
    :param alpha:
    :return: u_score
    """
    if(len(subset_indices)==0):
        return 0
    elif(len(subset_indices)==1):
        return 0
    else:
        f_acts = torch.tensor(itemgetter(*subset_indices)(final_activations))
        p_log_p = F.softmax(f_acts, dim=1) * F.log_softmax(f_acts, dim=1)
        H = -p_log_p.numpy()
        u_score  = np.sum(H,axis=1)
        return  u_score
    '''
    u_score = 0
    for index in subset_indices:
        f_acts = torch.tensor(final_activations[index])
        p_log_p = F.softmax(f_acts, dim=0) * F.log_softmax(f_acts, dim=0)
        score = (-1.0 * p_log_p.sum()).numpy()
        u_score += alpha * score
    return u_score
    '''


def compute_r_score(penultimate_activations, subset_indices, alpha=0.2):
    """
    Computes Redundancy Score: The point should be distant from all the other elements in the subset.
    :param penultimate_activations:
    :param subset_indices:
    :param alpha:
    :return:
    """
    #TODO: Make pdistajces efficient
    if(len(subset_indices)==0):
        return 0
    if(len(subset_indices)==1):
        return 0
    else:
        subset_p_acts = np.ndarray(itemgetter(*subset_indices)(penultimate_activations))
        pdist = np.sort(cdist(penultimate_activations,subset_p_acts),axis=1)
        r_score = pdist[:,1]
        print r_score
        return  r_score

    '''
    r_score = 0
    for index in subset_indices:
        p_act = penultimate_activations[index]
        if len(subset_indices) > 1:
            subset_penultimate_acts = [penultimate_activations[i] for i in subset_indices]
            dist = np.linalg.norm(subset_penultimate_acts - p_act, axis=1)
            score = np.min(dist[dist != 0])
            r_score += alpha * score
    return r_score
    '''


def compute_md_score(penultimate_activations, subset_indices, class_mean, alpha=2.):
    """
    Computes Mean Divergence score: The new datapoint should be close to the class mean
    :param penultimate_activations:
    :param subset_indices:
    :param class_mean:
    :param alpha:
    :return: list of scores for each subset item
    """
    if(len(subset_indices)==0):
        return 0
    elif(len(subset_indices)==1):
        return np.sqrt(np.sum(np.square(penultimate_activations[subset_indices[0]]-class_mean)))
    else:
        pen_act = np.array(itemgetter(*subset_indices)(penultimate_activations))-np.array(class_mean)
        md_score = alpha*np.sqrt(np.sum(np.square(pen_act),axis=1))
        #md_score = np.sum(np.square((pen_act)))
        #print md_score
        return md_score

    '''md_score = 0
    for index in subset_indices:
        p_act = penultimate_activations[index]
        score = np.linalg.norm(class_mean-p_act, axis=0)
        md_score += alpha * score
    return md_score'''
